package com.alibaba.cloud.ai.dashscope.audio.observation;

import io.micrometer.observation.ObservationConvention;

/**
 * Observation convention contract for DashScope audio transcription operations.
 */
public interface AudioTranscriptionObservationConvention
		extends ObservationConvention<AudioTranscriptionObservationContext> {

}
