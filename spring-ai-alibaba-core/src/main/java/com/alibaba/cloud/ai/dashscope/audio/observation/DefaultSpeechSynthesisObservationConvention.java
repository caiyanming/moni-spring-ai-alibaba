package com.alibaba.cloud.ai.dashscope.audio.observation;

import com.alibaba.cloud.ai.dashscope.audio.synthesis.SpeechSynthesisOptions;
import io.micrometer.common.KeyValue;
import io.micrometer.common.KeyValues;
import io.micrometer.observation.Observation.Context;
import org.springframework.util.StringUtils;

/**
 * Default observation convention for DashScope speech synthesis operations.
 */
public class DefaultSpeechSynthesisObservationConvention implements SpeechSynthesisObservationConvention {

	private static final String DEFAULT_NAME = "gen_ai.client.operation";

	private static final KeyValue REQUEST_MODEL_NONE = KeyValue
		.of(SpeechSynthesisObservationDocumentation.LowCardinalityKeyNames.REQUEST_MODEL, KeyValue.NONE_VALUE);

	@Override
	public boolean supportsContext(Context context) {
		return context instanceof SpeechSynthesisObservationContext;
	}

	@Override
	public String getName() {
		return DEFAULT_NAME;
	}

	@Override
	public String getContextualName(SpeechSynthesisObservationContext context) {
		SpeechSynthesisOptions options = context.getRequest().getOptions();
		if (options != null && StringUtils.hasText(options.getModel())) {
			return "%s %s".formatted(context.getOperationMetadata().operationType(), options.getModel());
		}
		return context.getOperationMetadata().operationType();
	}

	@Override
	public KeyValues getLowCardinalityKeyValues(SpeechSynthesisObservationContext context) {
		return KeyValues.of(operationType(context), provider(context), requestModel(context));
	}

	@Override
	public KeyValues getHighCardinalityKeyValues(SpeechSynthesisObservationContext context) {
		return KeyValues.empty();
	}

	protected KeyValue operationType(SpeechSynthesisObservationContext context) {
		return KeyValue.of(SpeechSynthesisObservationDocumentation.LowCardinalityKeyNames.AI_OPERATION_TYPE,
				context.getOperationMetadata().operationType());
	}

	protected KeyValue provider(SpeechSynthesisObservationContext context) {
		return KeyValue.of(SpeechSynthesisObservationDocumentation.LowCardinalityKeyNames.AI_PROVIDER,
				context.getOperationMetadata().provider());
	}

	protected KeyValue requestModel(SpeechSynthesisObservationContext context) {
		SpeechSynthesisOptions options = context.getRequest().getOptions();
		if (options != null && StringUtils.hasText(options.getModel())) {
			return KeyValue.of(SpeechSynthesisObservationDocumentation.LowCardinalityKeyNames.REQUEST_MODEL,
					options.getModel());
		}
		return REQUEST_MODEL_NONE;
	}

}
