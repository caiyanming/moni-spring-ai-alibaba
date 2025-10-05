package com.alibaba.cloud.ai.dashscope.audio.observation;

import io.micrometer.common.docs.KeyName;
import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationConvention;
import io.micrometer.observation.docs.ObservationDocumentation;
import org.springframework.ai.observation.conventions.AiObservationAttributes;

/**
 * Observation documentation for DashScope audio transcription operations.
 */
public enum AudioTranscriptionObservationDocumentation implements ObservationDocumentation {

	AUDIO_TRANSCRIPTION_OPERATION {
		@Override
		public Class<? extends ObservationConvention<? extends Observation.Context>> getDefaultConvention() {
			return DefaultAudioTranscriptionObservationConvention.class;
		}

		@Override
		public KeyName[] getLowCardinalityKeyNames() {
			return LowCardinalityKeyNames.values();
		}

		@Override
		public KeyName[] getHighCardinalityKeyNames() {
			return HighCardinalityKeyNames.values();
		}
	};

	public enum LowCardinalityKeyNames implements KeyName {

		AI_OPERATION_TYPE {
			@Override
			public String asString() {
				return AiObservationAttributes.AI_OPERATION_TYPE.value();
			}
		},
		AI_PROVIDER {
			@Override
			public String asString() {
				return AiObservationAttributes.AI_PROVIDER.value();
			}
		},
		REQUEST_MODEL {
			@Override
			public String asString() {
				return AiObservationAttributes.REQUEST_MODEL.value();
			}
		};

	}

	public enum HighCardinalityKeyNames implements KeyName {

		RESPONSE_ID {
			@Override
			public String asString() {
				return AiObservationAttributes.RESPONSE_ID.value();
			}
		};

	}

}
