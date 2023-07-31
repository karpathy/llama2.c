package com.celikin.llama2

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream


private fun userAssetPath(context: Context?): String {
    if (context == null)
        return ""
    val extDir = context.getExternalFilesDir("assets")
        ?: return context.getDir("assets", 0).absolutePath
    return extDir.absolutePath
}

fun Context.copyAssets(listFiles: Array<String>):String {
    val extFolder = userAssetPath(this)
    try {
        assets.list("")
            ?.filter { listFiles.contains(it) }
            ?.filter { !File(extFolder, it).exists() }
            ?.forEach {
                val target = File(extFolder, it)
                assets.open(it).use { input ->
                    FileOutputStream(target).use { output ->
                        input.copyTo(output)
                        Log.i("Utils", "Copied from apk assets folder to ${target.absolutePath}")
                    }
                }
            }
    } catch (e: Exception) {
        Log.e("Utils", "asset copy failed", e)
    }
    return extFolder
}