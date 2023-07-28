package com.celikin.llama2.wrapper

object InferenceRunner {

    private var inferenceCallback: InferenceCallback? = null

    fun setInferenceCallback(callback: InferenceCallback) {
        inferenceCallback = callback
    }

    external fun run(
        checkpoint: String,
        tokenizer: String,
        temperature: Float,
        steps: Int,
        prompt: String
    )

    external fun stop()

    fun onNewToken(token: String) {
        inferenceCallback?.onNewResult(token)
    }

    interface InferenceCallback {
        fun onNewResult(token: String?)
    }

    init {
        System.loadLibrary("inference")
    }

}