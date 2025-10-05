package com.alibaba.cloud.ai.dashscope.audio.observation;

import com.alibaba.cloud.ai.dashscope.common.DashScopeApiConstants;
import io.micrometer.common.KeyValue;
import io.micrometer.common.KeyValues;
import io.micrometer.observation.Observation.Context;
import org.springframework.ai.audio.transcription.AudioTranscriptionOptions;
import org.springframework.ai.audio.transcription.AudioTranscriptionResponse;
import org.springframework.ai.observation.conventions.AiObservationAttributes;
import org.springframework.util.StringUtils;

/**
 * Default observation convention for DashScope audio transcription operations.
 */
public class DefaultAudioTranscriptionObservationConvention implements AudioTranscriptionObservationConvention {

	private static final String DEFAULT_NAME = "gen_ai.client.operation";

	private static final KeyValue REQUEST_MODEL_NONE = KeyValue
		.of(AudioTranscriptionObservationDocumentation.LowCardinalityKeyNames.REQUEST_MODEL, KeyValue.NONE_VALUE);

	@Override
	public boolean supportsContext(Context context) {
		return context instanceof AudioTranscriptionObservationContext;
	}

	@Override
	public String getName() {
		return DEFAULT_NAME;
	}

	@Override
	public String getContextualName(AudioTranscriptionObservationContext context) {
		AudioTranscriptionOptions options = context.getRequest().getOptions();
		if (options != null && StringUtils.hasText(options.getModel())) {
			return "%s %s".formatted(context.getOperationMetadata().operationType(), options.getModel());
		}
		return context.getOperationMetadata().operationType();
	}

	@Override
	public KeyValues getLowCardinalityKeyValues(AudioTranscriptionObservationContext context) {
		return KeyValues.of(operationType(context), provider(context), requestModel(context));
	}

	@Override
	public KeyValues getHighCardinalityKeyValues(AudioTranscriptionObservationContext context) {
		KeyValues keyValues = KeyValues.empty();
		AudioTranscriptionResponse response = context.getResponse();
		if (response != null && response.getMetadata() != null) {
			Object requestId = response.getMetadata().get(AiObservationAttributes.RESPONSE_ID.value());
			if (requestId == null) {
				requestId = response.getMetadata().get(DashScopeApiConstants.REQUEST_ID);
			}
			if (requestId == null) {
				requestId = response.getMetadata().get("requestId");
			}
			if (requestId != null) {
				keyValues = keyValues.and(
						AudioTranscriptionObservationDocumentation.HighCardinalityKeyNames.RESPONSE_ID.asString(),
						requestId.toString());
			}
		}
		return keyValues;
	}

	protected KeyValue operationType(AudioTranscriptionObservationContext context) {
		return KeyValue.of(AudioTranscriptionObservationDocumentation.LowCardinalityKeyNames.AI_OPERATION_TYPE,
				context.getOperationMetadata().operationType());
	}

	protected KeyValue provider(AudioTranscriptionObservationContext context) {
		return KeyValue.of(AudioTranscriptionObservationDocumentation.LowCardinalityKeyNames.AI_PROVIDER,
				context.getOperationMetadata().provider());
	}

	protected KeyValue requestModel(AudioTranscriptionObservationContext context) {
		AudioTranscriptionOptions options = context.getRequest().getOptions();
		if (options != null && StringUtils.hasText(options.getModel())) {
			return KeyValue.of(AudioTranscriptionObservationDocumentation.LowCardinalityKeyNames.REQUEST_MODEL,
					options.getModel());
		}
		return REQUEST_MODEL_NONE;
	}

}
