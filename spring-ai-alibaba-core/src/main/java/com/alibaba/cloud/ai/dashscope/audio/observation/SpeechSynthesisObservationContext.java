package com.alibaba.cloud.ai.dashscope.audio.observation;

import com.alibaba.cloud.ai.dashscope.audio.synthesis.SpeechSynthesisPrompt;
import com.alibaba.cloud.ai.dashscope.audio.synthesis.SpeechSynthesisResponse;
import org.springframework.ai.model.observation.ModelObservationContext;
import org.springframework.ai.observation.AiOperationMetadata;

/**
 * Observation context for DashScope speech synthesis operations.
 */
public class SpeechSynthesisObservationContext
		extends ModelObservationContext<SpeechSynthesisPrompt, SpeechSynthesisResponse> {

	private static final String OPERATION_TYPE = "speech_synthesis";

	private SpeechSynthesisObservationContext(SpeechSynthesisPrompt prompt, String provider) {
		super(prompt, AiOperationMetadata.builder().operationType(OPERATION_TYPE).provider(provider).build());
	}

	public static Builder builder() {
		return new Builder();
	}

	public static final class Builder {

		private SpeechSynthesisPrompt prompt;

		private String provider;

		private Builder() {
		}

		public Builder prompt(SpeechSynthesisPrompt prompt) {
			this.prompt = prompt;
			return this;
		}

		public Builder provider(String provider) {
			this.provider = provider;
			return this;
		}

		public SpeechSynthesisObservationContext build() {
			return new SpeechSynthesisObservationContext(this.prompt, this.provider);
		}

	}

}
