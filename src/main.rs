use azure_cognitiveservices_speech::audio::AudioConfig;
use azure_cognitiveservices_speech::speech::{
    ResultReason, SpeechConfig, SpeechRecognizer, SpeechSynthesisOutputFormat,
};
use dotenv::dotenv;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> azure_cognitiveservices_speech::Result<()> {
    dotenv().ok();

    let speech_key = env::var("AZURE_SPEECH_KEY").expect("AZURE_SPEECH_KEY must be set");
    let service_region = env::var("AZURE_SERVICE_REGION").expect("AZURE_SERVICE_REGION must be set");
    let audio_filename = env::var("SOUND_FILE").expect("SOUND_FILE must be set");
    let output_markdown_filename = env::var("OUTPUT_FILE").expect("OUTPUT_FILE must be set");

    let speech_config = SpeechConfig::from_subscription(&speech_key, &service_region)?;
    speech_config.set_speech_recognition_language("en-US")?;
    speech_config.request_word_level_timestamps()?;
    speech_config.enable_dictation()?;
    speech_config.set_output_format(SpeechSynthesisOutputFormat::DetailedJson)?;

    let audio_config = AudioConfig::from_wav_file_input(&audio_filename)?;
    let recognizer = SpeechRecognizer::new(speech_config, Some(audio_config))?;

    let transcript_data = Arc::new(Mutex::new(Vec::new()));
    let speakers = Arc::new(Mutex::new(HashMap::new()));

    recognizer.recognized.connect({
        let transcript_data = Arc::clone(&transcript_data);
        let speakers = Arc::clone(&speakers);
        move |event| {
            if let ResultReason::RecognizedSpeech = event.result.reason {
                let result_json = event.result.priv_text.as_ref().unwrap();
                let result_json: serde_json::Value = serde_json::from_str(result_json).unwrap();

                let mut locked_transcript = transcript_data.lock().await;
                locked_transcript.push(result_json.clone());

                if let Some(nbest) = result_json.get("NBest").and_then(|n| n.as_array()) {
                    for sentence in nbest.iter().flat_map(|s| s.get("Words").and_then(|w| w.as_array())) {
                        for word in sentence {
                            let speaker_id = word.get("SpeakerId").and_then(|s| s.as_str()).unwrap_or("Unknown");
                            let mut locked_speakers = speakers.lock().await;
                            let speaker = locked_speakers
                                .entry(speaker_id.to_string())
                                .or_insert_with(|| format!("Speaker {}", locked_speakers.len() + 1))
                                .clone();
                            let text = word.get("Word").and_then(|w| w.as_str()).unwrap();
                            let start_time = word.get("Offset").and_then(|o| o.as_f64()).unwrap() / 10_000_000.0;
                            let end_time = (word.get("Offset").and_then(|o| o.as_f64()).unwrap()
                                + word.get("Duration").and_then(|d| d.as_f64()).unwrap())
                                / 10_000_000.0;

                            println!("- **{}** ({:.2}s - {:.2}s): {}", speaker, start_time, end_time, text);
                        }
                    }
                }
            }
        }
    }).await;

    println!("Starting transcription and diarization...");
    recognizer.start_continuous_recognition().await?;

    while !recognizer.session_started().await {}
    recognizer.stop_continuous_recognition().await?;

    println!("Transcription completed. Writing to output file...");
    let mut output_file = File::create(&output_markdown_filename)?;

    let locked_transcript = transcript_data.lock().await;
    for result in locked_transcript.iter() {
        if let Some(nbest) = result.get("NBest").and_then(|n| n.as_array()) {
            for sentence in nbest {
                if let Some(words) = sentence.get("Words").and_then(|w| w.as_array()) {
                    for word in words {
                        let speaker_id = word.get("SpeakerId").and_then(|s| s.as_str()).unwrap_or("Unknown");
                        let locked_speakers = speakers.lock().await;
                        let speaker = locked_speakers.get(speaker_id).unwrap();
                        let text = word.get("Word").and_then(|w| w.as_str()).unwrap();
                        let start_time = word.get("Offset").and_then(|o| o.as_f64()).unwrap() / 10_000_000.0;
                        let end_time = (word.get("Offset").and_then(|o| o.as_f64()).unwrap()
                            + word.get("Duration").and_then(|d| d.as_f64()).unwrap())
                            / 10_000_000.0;

                        writeln!(output_file, "- **{}** ({:.2}s - {:.2}s): {}", speaker, start_time, end_time, text)?;
                    }
                }
            }
        }
    }

    println!("Output written to {}", output_markdown_filename);
    Ok(())
}