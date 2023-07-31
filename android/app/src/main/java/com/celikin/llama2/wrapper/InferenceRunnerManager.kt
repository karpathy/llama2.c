package com.celikin.llama2.wrapper


import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class InferenceRunnerManager {
    private lateinit var folderPath: String
    private val applicationScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    fun init(callback: InferenceRunner.InferenceCallback, folderPath: String) {
        this.folderPath = folderPath
        InferenceRunner.setInferenceCallback(callback)
    }

    fun run(
        prompt: String = "",
        temperature: Float = 0.9f,
        steps: Int = 256,
        checkpointFileName: String = "stories15M.bin",
        tokenizerFileName: String = "tokenizer.bin",
        ompThreads: Int = 4,
    ) {
        applicationScope.launch {
            InferenceRunner.run(
                checkpoint = "$folderPath/$checkpointFileName",
                tokenizer = "$folderPath/$tokenizerFileName",
                temperature = temperature,
                steps = steps,
                prompt = prompt,
                ompthreads = ompThreads
            )
        }
    }
}