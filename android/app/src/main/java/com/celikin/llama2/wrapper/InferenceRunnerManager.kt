package com.celikin.llama2.wrapper


import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class InferenceRunnerManager {
    private val applicationScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    fun init(callback: InferenceRunner.InferenceCallback) {
        InferenceRunner.setInferenceCallback(callback)
    }

    fun run(prompt: String = "") {
        applicationScope.launch {
            InferenceRunner.run(
                checkpoint = "/data/local/tmp/stories15M.bin",
                tokenizer = "/data/local/tmp/tokenizer.bin",
                temperature = 0.9f,
                steps = 256,
                prompt = prompt
            )
        }
    }
}