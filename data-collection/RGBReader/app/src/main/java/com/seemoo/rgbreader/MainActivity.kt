package com.seemoo.rgbreader

import android.Manifest
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.usb.UsbAccessory
import android.hardware.usb.UsbManager
import android.media.MediaScannerConnection.scanFile
import android.net.Uri
import android.os.Bundle
import android.os.ParcelFileDescriptor
import android.os.PowerManager
import android.util.Log
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.covertbagel.androidopenaccessorybridge.AndroidOpenAccessoryBridge
import com.covertbagel.androidopenaccessorybridge.BufferHolder
import com.instacart.library.truetime.TrueTime
import com.seemoo.rgbreader.AccessoryEngine.IEngineCallback
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.*
import java.lang.Math.abs
import java.nio.charset.Charset
import java.text.SimpleDateFormat
import java.util.*
import kotlin.concurrent.thread
import kotlin.text.Charsets.UTF_8


enum class Color (val rgb: Int) {
    RED(0),
    GREEN(1),
    BLUE(2),
    DAYLIGHT(3),
}

class MainActivity : AppCompatActivity() {

    val TAG = "RGBReader"
    /*
    USB Stuff
     */

    private var device : UsbAccessory? = null
    private var manager : UsbManager?  = null
    private val context : MainActivity = this
    private var fileDescriptor: ParcelFileDescriptor? = null
    private var inputStream: FileInputStream? = null
    private var outputStream: FileOutputStream? = null
    private var watchThread:Thread = Thread()
    private var mEngine: AccessoryEngine? = null



    private val ACTION_USB_PERMISSION = "com.seemoo.rgbreader.USB_PERMISSION"

    val mCallback_engine: IEngineCallback = object: IEngineCallback{
        override fun onDeviceDisconnected() {
            println("device Disconnected")
        }

        override fun onConnectionEstablished() {
            println("device connected")
        }

        override fun onConnectionClosed() {
            println("Connection closed")
        }
        override fun onDataRecieved(data: ByteArray?, num: Int) {
            var str:String = "data recvied"
            str = String(data!!).filter { it.isLetterOrDigit() }
            Log.d(TAG, "onDataRecieved: "+str)
            println(str)
            checkCommand(str)
        }

    }

    val receiver = object :  BroadcastReceiver(){
        override fun onReceive(context: Context, intent: Intent) {
            println("Got Boradcast")
            if (ACTION_USB_PERMISSION == intent.action) {
                synchronized(this) {
                    val accessory: UsbAccessory? = intent.getParcelableExtra(UsbManager.EXTRA_ACCESSORY)

                    if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)) {
                        accessory?.apply {
                            //call method to set up accessory communication
                            println("Accessorry accespted")
                            fileDescriptor = manager?.openAccessory(accessory)
                            fileDescriptor?.fileDescriptor.also { fd ->
                                inputStream = FileInputStream(fd)
                                print("did it work ?")
                                outputStream = FileOutputStream(fd)
                                watchThread = Thread(null,mCallback,"accessory")
                                watchThread.start()
                            }

                        }
                    } else {
                        println("permission denied")
                    }
                }
            }
        }

    }



    var sb : StringBuilder = StringBuilder()
    val mCallback:Runnable = Runnable { ->
        var data:ByteArray = ByteArray(8*1024)
        try {
            var data = inputStream?.readBytes()
        }catch (e: IOException) {
            Log.d(TAG, "error: " + e.toString())
        }
        var str = ""
        if (data != null) str = String(data)
        println("got Data "+str)
        if (str.contains("\n")) {
            str = str.replace("\n", "")
            sb.append(str)
            checkCommand(sb.toString())
            sb = StringBuilder()

        } else {
            sb.append(str)
        }

    }

    private fun checkCommand(toString: String) {
        if (toString == "start") {
            println("checkcommand Activated")
            runOnUiThread{ -> startRecoding(View(this))}
            // just pass a view its not used anyway)

        } else if (toString == "stop"){
            runOnUiThread { -> startRecoding(View(this)) }
        }
    }


    /*
    Sensor Stuff
     */
    private val NUM_DETECTIONS = 9
    private val permissionCode: Int = 111
    private val ntpServer:String = "ptbtime1.ptb.de"
    private lateinit var sensorManager : SensorManager
    private lateinit var fileWriter : File
    private var rgbSensor:Sensor? = null
    val currentLocale = Locale.getDefault()
    private val simpleDate = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS",currentLocale)
    private val fileDate = SimpleDateFormat("yyyy_MM_dd_HH_mm_ss",currentLocale)
    private var recording = false
    private lateinit var interpreter: Interpreter
    private lateinit var savedir : File
    private lateinit var aoa:AndroidOpenAccessoryBridge
    var listener = object: SensorEventListener{
        override fun onAccuracyChanged(p0: Sensor?, p1: Int) {
            Log.d("Recorder", "Accuracy Changed: $p1")
        }

        override fun onSensorChanged(p0: SensorEvent?) {
            val time = TrueTime.now()
            writeToFile(p0?.values,time)
            rgbChecker(p0?.values)
        }

    }
    lateinit var wakelock : PowerManager.WakeLock



    private fun rgbChecker(values: FloatArray?) {
        //enum = {"red":0,"green":1,"blue":2,"daylight":3}
        val color_number = values?.get(1)
        val b = abs(400000- color_number!!)
        val day = abs(4100 -color_number!!)
        val g = abs(9800 - color_number!!)
        val r = abs(1900 -color_number!!)
        val result = listOf<Float>(r,g,b,day).min()
        if (result == b){
            if (!textView.text.equals(getString(R.string.blue))) textView.setText(R.string.blue)
        } else if(result == day){
            if (!textView.text.equals(getString(R.string.daylight))) textView.setText(R.string.daylight)
        }else if(result == g){
            if (!textView.text.equals(getString(R.string.green))) textView.setText(R.string.green)
        }else if(result == r){
            if (!textView.text.equals(getString(R.string.red))) textView.setText(R.string.red)
        }
        val output = TensorBuffer.createFixedSize(intArrayOf(1,4),DataType.FLOAT32)
        val input = TensorBuffer.createFixedSize(intArrayOf(1,3),DataType.FLOAT32)
        input.loadArray(floatArrayOf(values[0], values[1], values[2]))
        interpreter.run(input.buffer,output.buffer)
        val res = output.floatArray
        val color = Color.values().get(res.indexOf(res.max()!!))
        textView2.setText(color.name)
        Log.d("Classified",color.name)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        rgbSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT)
        reqPermissions()
        val f = File(cacheDir,"converted_model.tflite")
        val instream = assets.open("converted_model.tflite")
        wakelock = (getSystemService(Context.POWER_SERVICE)as PowerManager).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK,"myapp::cpu_wakelock")
        copyStreamToFile(instream,f)
        interpreter = Interpreter(f)
        val tensor = interpreter.getInputTensor(0)
        Log.d("Classifier",tensor.shape().toString())
        //initUSBConnection()
        onNewIntent(intent)

    }

    override fun onNewIntent(intent: Intent?) {
        if (mEngine == null){
            mEngine = AccessoryEngine(this,mCallback_engine)
        }
        mEngine!!.onNewIntent(intent)
        super.onNewIntent(intent)
    }

    private fun initUSBConnection() {
        manager = getSystemService(Context.USB_SERVICE) as UsbManager
        Thread{
            manager = getSystemService(Context.USB_SERVICE) as UsbManager
            while (true) {
                val list = manager!!.accessoryList
                if (list != null) {
                    for (entry in list) {
                        var pi =
                            PendingIntent.getBroadcast(context, 0, Intent(ACTION_USB_PERMISSION), 0)
                        device = entry
                        println("request permission")
                        manager!!.requestPermission(entry, pi)
                        break
                    }
                    if (device != null) {
                        break
                    }
                }
                Thread.sleep(500)
            }
        }.start()

        val filter = IntentFilter(ACTION_USB_PERMISSION)
        filter.addAction(UsbManager.ACTION_USB_ACCESSORY_DETACHED)
        registerReceiver(receiver,filter)

    }

    private fun println(str :String){
        runOnUiThread {  ->
            textview_usb.text = str
        }
    }

    private fun copyStreamToFile(instream: InputStream, f: File) {
        instream.use { input ->
            val outputStream = FileOutputStream(f)
            outputStream.use { output ->
                val buffer = ByteArray(4*1024)
                while (true){
                    val bytecount = input.read(buffer)
                    if (bytecount == -1) break
                    output.write(buffer)
                }
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        var i: Int = 0;
        for (p in permissions){
            if (p.equals(Manifest.permission.INTERNET) && grantResults[i] == PackageManager.PERMISSION_GRANTED){
                initTime()
            }
            if (p.equals(Manifest.permission.WRITE_EXTERNAL_STORAGE) && grantResults[i] == PackageManager.PERMISSION_GRANTED){
                initPath()
            }
            i++
        }
        recordBtn.isEnabled = true

    }

    private fun initPath() {
        var path = this.getExternalFilesDir(null)
        path = File(path?.path?.split("Android")?.get(0))
        savedir = File(path,"SensorRecordings")
        savedir.mkdirs()
    }

    private fun initTime() {
        val t = thread(true) {
            TrueTime.build().withNtpHost(ntpServer).initialize()
            Log.d("Recorder", "NTP Server Ready")
        }
        t.join()
    }

    private fun reqPermissions() {
        ActivityCompat.requestPermissions(this,arrayOf(Manifest.permission.INTERNET,Manifest.permission.WRITE_EXTERNAL_STORAGE),permissionCode)

    }

    private var t : Timer = Timer()

    fun startRecoding(view: View) {
        if (recording){
            stopRecording()
        } else{
            wakelock.acquire()
            var i = 0;
            var str = ""
            str = i.toString()
            while (i < 5) {
                //Thread.sleep(1000)
                i++
            }
            fileWriter = File(savedir,"recording_"+ fileDate.format(TrueTime.now()) +".txt")
            sensorManager.registerListener(listener,rgbSensor,SensorManager.SENSOR_DELAY_FASTEST)
            recordBtn.setText(getString(R.string.recording))
            t = Timer()
            t.schedule(object: TimerTask(){
               var i =  0
                override fun run() {
                    if (i == 9) {
                        runOnUiThread {
                            stopRecording()

                        }

                        i = 0
                    } else i++
                }
            },60*1000)
            t.schedule(object: TimerTask() {
                var time = 0
                override fun run() {
                    runOnUiThread { timerText.setText(timeToString(++time)) }
                }
            },1000,1000)

        }
        recording = !recording
    }

    private fun timeToString(i: Int): String {
        if (i == 0) return "0:0"
        val sb = StringBuilder()
        sb.append((i/60))
        sb.append(":")
        sb.append(i % 60)
        return sb.toString()
    }

    private fun stopRecording() {
        t.cancel()
        t.purge()
        sensorManager.unregisterListener(listener)
        recordBtn.setText(getString(R.string.start))
        this.importFile()
        wakelock.release()
    }

    private fun importFile() {
        scanFile(this, arrayOf(fileWriter.absolutePath),null,{path: String?, uri: Uri? -> Log.d("Recorder",
            "Scan Complete $path"
        ) })
    }

    private fun writeToFile(values: FloatArray?, time: Date?) {
        if (mEngine != null){
            mEngine!!.write(stringify(values,time).toByteArray(java.nio.charset.StandardCharsets.UTF_8))
        }
        fileWriter.appendText(stringify(values,time))
        Log.d("Recorder",stringify(values,time))
    }

    private fun stringify(values: FloatArray?, time: Date?): String {
        val str = StringBuilder()
        str.append(simpleDate.format(time))
        str.append(" ")
        str.append(values?.get(0))
        str.append(", ")
        str.append(values?.get(1))
        str.append(", ")
        str.append(values?.get(2))
        str.append("\n")
        return str.toString()
    }


}
