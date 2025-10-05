package com.alibaba.cloud.ai.dashscope.audio.observation;

import org.springframework.ai.audio.transcription.AudioTranscriptionPrompt;
import org.springframework.ai.audio.transcription.AudioTranscriptionResponse;
import org.springframework.ai.model.observation.ModelObservationContext;
import org.springframework.ai.observation.AiOperationMetadata;

/**
 * Observation context for DashScope audio transcription operations.
 */
public class AudioTranscriptionObservationContext
		extends ModelObservationContext<AudioTranscriptionPrompt, AudioTranscriptionResponse> {

	private static final String OPERATION_TYPE = "audio_transcription";

	private AudioTranscriptionObservationContext(AudioTranscriptionPrompt prompt, String provider) {
		super(prompt, AiOperationMetadata.builder().operationType(OPERATION_TYPE).provider(provider).build());
	}

	public static Builder builder() {
		return new Builder();
	}

	public static final class Builder {

		private AudioTranscriptionPrompt prompt;

		private String provider;

		private Builder() {
		}

		public Builder prompt(AudioTranscriptionPrompt prompt) {
			this.prompt = prompt;
			return this;
		}

		public Builder provider(String provider) {
			this.provider = provider;
			return this;
		}

		public AudioTranscriptionObservationContext build() {
			return new AudioTranscriptionObservationContext(this.prompt, this.provider);
		}

	}

}
