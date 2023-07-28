package com.celikin.llama2

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.celikin.llama2.databinding.ActivityMainBinding
import com.celikin.llama2.wrapper.InferenceRunner
import com.celikin.llama2.wrapper.InferenceRunnerManager

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var inferenceRunnerManager: InferenceRunnerManager

    private val callback = object : InferenceRunner.InferenceCallback {
        override fun onNewResult(token: String?) {
            token?.let { runOnUiThread { updateText(token) } }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.regenerate.setOnClickListener {
            binding.sampleText.text = ""
            val prompt = binding.promptEdit.text.toString()
            inferenceRunnerManager.run(prompt)
        }
        initInference()
    }

    private fun updateText(token: String) {
        binding.sampleText.text = "${binding.sampleText.text}$token"
    }

    private fun initInference() {
        inferenceRunnerManager = InferenceRunnerManager().apply { init(callback) }
    }

}