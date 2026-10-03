//! The core API a consumer sees without the VST3/CLAP plugin layer.
//!
//! Integration tests are compiled as their own crate against the library's
//! public surface, so a `pub` that regresses to `pub(crate)` fails here rather
//! than in a downstream project. Runs in every feature shape, including the
//! plugin-less default build.

use deepfilter_vst::{
    AudioChunk, AudioChunkError, DfEngine, DspCore, ModelError, ModelInfo, ModelSource,
    RateError, RatePlan, WorkerError, WorkerHandle, MAX_HOST_QUANTUM, MODEL_HOP_SIZE,
    MODEL_PATH_ENV, MODEL_SAMPLE_RATE,
};

#[test]
fn the_core_types_are_nameable_from_outside_the_crate() {
    fn nameable<T>() {}
    nameable::<DfEngine>();
    nameable::<DspCore>();
    nameable::<WorkerHandle>();
    nameable::<ModelSource<'static>>();
    nameable::<ModelInfo>();
    nameable::<ModelError>();
    nameable::<RateError>();
    nameable::<WorkerError>();

    assert_eq!(MODEL_PATH_ENV, "DEEPFILTER_MODEL");
    assert_eq!(MODEL_SAMPLE_RATE, 48_000);
    assert_eq!(MODEL_HOP_SIZE, 480);
    assert_eq!(MAX_HOST_QUANTUM, 1_920);
}

#[test]
fn the_rate_plan_and_transport_types_are_public() {
    let plan = RatePlan::preflight(MODEL_SAMPLE_RATE).expect("48 kHz must preflight");
    assert_eq!(plan.host_sample_rate, MODEL_SAMPLE_RATE);
    assert_eq!(plan.host_quantum, MODEL_HOP_SIZE);

    let samples = vec![0.25_f32; plan.host_quantum];
    let mut chunk = AudioChunk::from_slice(1, 0, &samples).expect("one quantum must fit");
    assert_eq!(chunk.generation(), 1);
    assert_eq!(chunk.start_sample(), 0);
    assert_eq!(chunk.len(), plan.host_quantum);
    assert!(!chunk.is_empty());
    assert_eq!(chunk.samples(), samples.as_slice());

    chunk.set_generation(2);
    assert_eq!(chunk.generation(), 2);

    let oversized = [0.0; MAX_HOST_QUANTUM + 1];
    assert!(matches!(
        AudioChunk::from_slice(0, 0, &oversized),
        Err(AudioChunkError::TooLong)
    ));
    assert!(AudioChunkError::TooLong.to_string().contains("1920"));
}
