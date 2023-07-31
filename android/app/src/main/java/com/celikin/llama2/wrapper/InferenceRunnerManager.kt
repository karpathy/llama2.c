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

    fun run(
        prompt: String = "",
        temperature: Float = 0.9f,
        steps: Int = 256,
        checkpoint: String = "/data/local/tmp/stories15M.bin",
        tokenizer: String = "/data/local/tmp/tokenizer.bin",
        ompthreads: Int = 4,
    ) {
        applicationScope.launch {
            InferenceRunner.run(
                checkpoint = checkpoint,
                tokenizer = tokenizer,
                temperature = temperature,
                steps = steps,
                prompt = prompt,
                ompthreads = ompthreads
            )
        }
    }
}