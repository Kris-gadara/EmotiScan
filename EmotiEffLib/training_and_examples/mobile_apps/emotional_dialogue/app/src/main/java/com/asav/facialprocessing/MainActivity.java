package com.asav.facialprocessing;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.view.MenuCompat;
import android.text.method.ScrollingMovementMethod;

import android.content.*;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.*;
import android.media.Image;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.speech.*;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.*;

import com.asav.facialprocessing.mtcnn.Box;
import com.asav.facialprocessing.mtcnn.MTCNN;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;
import androidx.camera.core.*;

public class MainActivity extends AppCompatActivity{
    private static final String TAG = "MainActivity";
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;
    private ImageView imageView;
    private Bitmap sampledImage=null;
    private TextView textViewUserRequest,textViewResponse;

    private HandlerThread mBackgroundThread=null;
    private Handler mBackgroundHandler=null;

    private static int MIN_FACE_SIZE = 64;
    private static double MIN_VIDEO_SIZE = 480.0;
    private MTCNN mtcnnFaceDetector=null;
    private EmotionPyTorchClassifier emotionClassifierPyTorch = null;
    private String currentEmotion="";

    private SpeechRecognizer mSpeech = null;
    private Intent mIntent;

    private ServerProcessor serverProcessor=null;

    private EditText serverView=null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        androidx.appcompat.widget.Toolbar toolbar = (androidx.appcompat.widget.Toolbar) findViewById(R.id.main_toolbar);
        setSupportActionBar(toolbar);
        imageView=(ImageView)findViewById(R.id.inputImageView);
        textViewUserRequest = findViewById(R.id.textUserRequest);
        textViewResponse= findViewById(R.id.textResponse);
        textViewResponse.setMovementMethod(new ScrollingMovementMethod());
        serverView = findViewById(R.id.server_address);

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, getRequiredPermissions(), REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS);
        }
        else
            init();
    }
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.toolbar_menu, menu);
        MenuCompat.setGroupDividerEnabled(menu, true);
        return true;
    }

    private void init(){
        /*new Thread(() -> {
                try{
                //String question="suggestion of implementing emotional answers in a personalized assistant";
                //processRequest(question,currentEmotion);
                HttpServerProcessor client=new HttpServerProcessor(this);
                } catch (final Exception e) {
                    Log.e(TAG, "Exception initializing HttpServerProcessor!", e);
                }
            }).start();*/
        try {
            emotionClassifierPyTorch =new EmotionPyTorchClassifier(getApplicationContext());
        } catch (final Exception e) {
            Log.e(TAG, "Exception initializing EmotionPyTorchClassifier!", e);
        }
        try {
            mtcnnFaceDetector =new MTCNN(getApplicationContext());
            //mtcnnFaceDetector = MTCNNModel.Companion.create(getAssets());
        } catch (final Exception e) {
            Log.e(TAG, "Exception initializing MTCNNModel!"+e);
        }

        try{
            mSpeech = SpeechRecognizer.createSpeechRecognizer(this.getApplicationContext());
            mIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
            //mIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
            mIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.US.toString());

            mSpeech.setRecognitionListener(new RecognitionListener() {
                @Override
                public void onReadyForSpeech(Bundle bundle) {
                    Log.v("InsideOutSpeechRecognizer","onReadyForSpeech");
                    //Toast.makeText(MainActivity.this, "Speech Recognition ready", Toast.LENGTH_SHORT).show();
                }

                @Override
                public void onBeginningOfSpeech() {
                    Log.v("InsideOutSpeechRecognizer","onBeginningOfSpeech");
                }

                @Override
                public void onRmsChanged(float v) {
                    //Log.v("InsideOutSpeechRecognizer","onRmsChanged:" + v);
                    /*
                    if ((new Random()).nextInt(2) == 1) {
                        textView.setBackgroundColor(Color.LTGRAY);
                    }
                    else {
                        textView.setBackgroundColor(Color.GRAY);
                    }*/
                }

                @Override
                public void onBufferReceived(byte[] bytes) {
                    Log.v("InsideOutSpeechRecognizer","onBufferReceived: " + bytes.length);
                }

                @Override
                public void onEndOfSpeech() {
                    Log.v("InsideOutSpeechRecognizer","onEndOfSpeech");
                    //textView.setBackgroundColor(Color.WHITE);
                }

                @Override
                public void onError(int i) {
                    //http://developer.android.com/reference/android/speech/SpeechRecognizer.html#ERROR_INSUFFICIENT_PERMISSIONS
                    Log.v("InsideOutSpeechRecognizer","error: " + i);
                    //textView.setBackgroundColor(Color.WHITE);
                }

                @Override
                public void onResults(Bundle bundle) {
                    ArrayList<String> words = bundle.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                    float[] scores = bundle.getFloatArray(android.speech.SpeechRecognizer.CONFIDENCE_SCORES);
                    String recognitionResult=words.get(0);
                    Log.v("InsideOutSpeechRecognizer","Recognition results first word: "+recognitionResult);
                    //Log.v("myapp",scores[0]));
                    /*String listString = String.join(", ", words);
                    Log.i("InsideOutSpeechRecognizer","Recognition results all words: "+listString);*/

                    textViewUserRequest.setText("Current emotion: "+currentEmotion+"\nRequest: "+recognitionResult);
                    //textView.setBackgroundColor(Color.WHITE);

                    new Thread(() -> {
                        //String question="suggestion of implementing emotional answers in a personalized assistant";
                        processRequest(recognitionResult,currentEmotion);
                    }).start();

                }

                @Override
                public void onPartialResults(Bundle bundle) {
                    Log.v("InsideOutSpeechRecognizer","onPartialResults");
                }

                @Override
                public void onEvent(int i, Bundle bundle) {
                    Log.v("InsideOutSpeechRecognizer","onEvent:" + i);

                }
            });
        }catch (final Exception e) {
            Log.e(TAG, "Exception initializing SpeechRecognizer!"+e);
        }

    }


    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.action_capturecamera:
                if(mBackgroundThread==null){
                    item.setTitle(R.string.action_StopRecognition);
                    setupCameraSpeech();
                }
                else{
                    item.setTitle(R.string.action_StartRecognition);
                    stopCameraSpeech();
                }
                return true;
            default:
                // If we got here, the user's action was not recognized.
                // Invoke the superclass to handle it.
                return super.onOptionsItemSelected(item);
        }
    }

    private static Map<String,Integer> emotion2color=Map.of("happy",Color.GREEN,
            "sad",Color.BLUE,
            "fear",Color.MAGENTA,
            "angry",Color.RED);
    //colors=[(0,128,0),(128,0,0),(128, 0, 128),(0,0,255)] //bgr

    private void processRequest(String question,String user_emotion){
        String serverAddress=serverView.getText().toString();

        if(serverProcessor==null || serverProcessor.getServerAddress()!=serverAddress){
            serverProcessor=new ServerProcessor(this,serverAddress);
        }
        if(serverProcessor!=null){
            long startTime = SystemClock.uptimeMillis();

            Map<String,String> responses=null;
            if(false){
                responses=serverProcessor.insideout(question,user_emotion);
            }
            else{
                responses=serverProcessor.single_call(question,user_emotion);
            }

            long timecostMs=SystemClock.uptimeMillis() - startTime;
            Log.i(TAG, "Timecost to run server inference: " + timecostMs+" "+responses);
            for(Map.Entry<String,String> e:responses.entrySet()){
                String emotion=e.getKey();
                runOnUiThread(()->{
                    textViewResponse.setText(emotion+":"+e.getValue());
                    if(emotion2color.containsKey(emotion))
                        textViewResponse.setTextColor(emotion2color.get(emotion));
                    else
                        textViewResponse.setTextColor(Color.BLACK);
                });
                try {
                    Thread.sleep(5000);
                } catch (InterruptedException ex) {
                    Log.e(TAG, "Exception sleep!", ex);
                }
            }
        }
    }
    private void setupCameraSpeech() {
        if(false){
            new Thread(() -> {
                String question="suggestion of implementing emotional answers in a personalized assistant";
                processRequest(question,currentEmotion);
            }).start();
            return;
        }
        serverView.clearFocus();
        PreviewConfig previewConfig = new PreviewConfig.Builder()
                .setLensFacing(CameraX.LensFacing.FRONT)
                .build();
        Preview preview = new Preview(previewConfig);
        mBackgroundThread = new HandlerThread("AnalysisThread");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());

        ImageAnalysis imageAnalysis = new ImageAnalysis(new ImageAnalysisConfig.Builder()
                .setLensFacing(CameraX.LensFacing.FRONT)
                .setCallbackHandler(mBackgroundHandler)
                .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
                .build());
        imageAnalysis.setAnalyzer(
                new ImageAnalysis.Analyzer() {
                    public void analyze(ImageProxy image, int rotationDegrees) {
                        sampledImage=imgToBitmap(image.getImage(), rotationDegrees);
                        recognizeEmotions(MIN_VIDEO_SIZE);
                    }
                }
        );

        CameraX.unbindAll();
        CameraX.bindToLifecycle(this, preview, imageAnalysis);

        mSpeech.startListening(mIntent);
    }
    private void stopCameraSpeech() {
        CameraX.unbindAll();
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
        } catch (InterruptedException e) {
            Log.e(TAG, "Exception stoppingCamera!", e);
        }
        mBackgroundThread = null;
        mBackgroundHandler = null;

        mSpeech.stopListening();
    }
    private boolean isCameraRunning(){
        if(mBackgroundThread!=null)
            Toast.makeText(getApplicationContext(),
                    "Stop camera firstly",
                    Toast.LENGTH_SHORT).show();
        return mBackgroundThread!=null;
    }

    private void setImage(Bitmap bitmap){
        runOnUiThread(new Runnable() {

            @Override
            public void run() {
                imageView.setImageBitmap(bitmap);
            }
        });
    }

    private void recognizeEmotions(double minSize){
        Bitmap bmp = sampledImage;
        Bitmap resizedBitmap=bmp;
        double scale=Math.min(bmp.getWidth(),bmp.getHeight())/minSize;
        if(scale>1.0) {
            resizedBitmap = Bitmap.createScaledBitmap(bmp, (int)(bmp.getWidth()/scale), (int)(bmp.getHeight()/scale), false);
            bmp=resizedBitmap;
        }
        long startTime = SystemClock.uptimeMillis();
        Vector<Box> bboxes = mtcnnFaceDetector.detectFaces(resizedBitmap, MIN_FACE_SIZE);//(int)(bmp.getWidth()*MIN_FACE_SIZE));
        Log.i(TAG, "Timecost to run mtcnn: " + Long.toString(SystemClock.uptimeMillis() - startTime));

        Bitmap tempBmp = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(tempBmp);
        Paint p = new Paint();
        p.setStyle(Paint.Style.STROKE);
        p.setAntiAlias(true);
        p.setFilterBitmap(true);
        p.setDither(true);
        p.setColor(Color.BLUE);
        p.setStrokeWidth(5);

        Paint p_text = new Paint();
        p_text.setColor(Color.WHITE);
        p_text.setStyle(Paint.Style.FILL);
        p_text.setColor(Color.BLUE);
        p_text.setTextSize(24);

        c.drawBitmap(bmp, 0, 0, null);

        for (Box box : bboxes) {
            android.graphics.Rect bbox = box.transform2Rect();//new android.graphics.Rect(Math.max(0,box.left()),Math.max(0,box.top()),box.right(),box.bottom());
            p.setColor(Color.RED);
            c.drawRect(bbox, p);
            if(emotionClassifierPyTorch!=null && bbox.width()>0 && bbox.height()>0) {
                int w=bmp.getWidth();
                int h=bmp.getHeight();
                android.graphics.Rect bboxOrig = new android.graphics.Rect(
                        Math.max(0,w*bbox.left / resizedBitmap.getWidth()),
                        Math.max(0,h*bbox.top / resizedBitmap.getHeight()),
                        Math.min(w,w * bbox.right / resizedBitmap.getWidth()),
                        Math.min(h,h * bbox.bottom / resizedBitmap.getHeight())
                );
                Bitmap faceBitmap = Bitmap.createBitmap(bmp, bboxOrig.left, bboxOrig.top, bboxOrig.width(), bboxOrig.height());
                currentEmotion=emotionClassifierPyTorch.recognize(faceBitmap);
                c.drawText(currentEmotion, Math.max(0,bbox.left), Math.max(0, bbox.top - 20), p_text);
                Log.i(TAG, currentEmotion);
            }
        }
        setImage(tempBmp);
    }

    private Bitmap imgToBitmap(Image image, int rotationDegrees) {
        // NV21 is a plane of 8 bit Y values followed by interleaved  Cb Cr
        ByteBuffer ib = ByteBuffer.allocate(image.getHeight() * image.getWidth() * 2);

        ByteBuffer y = image.getPlanes()[0].getBuffer();
        ByteBuffer cr = image.getPlanes()[1].getBuffer();
        ByteBuffer cb = image.getPlanes()[2].getBuffer();
        ib.put(y);
        ib.put(cb);
        ib.put(cr);

        YuvImage yuvImage = new YuvImage(ib.array(),
                ImageFormat.NV21, image.getWidth(), image.getHeight(), null);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0,
                image.getWidth(), image.getHeight()), 50, out);
        byte[] imageBytes = out.toByteArray();
        Bitmap bm = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
        Bitmap bitmap = bm;

        // On android the camera rotation and the screen rotation
        // are off by 90 degrees, so if you are capturing an image
        // in "portrait" orientation, you'll need to rotate the image.
        if (rotationDegrees != 0) {
            Matrix matrix = new Matrix();
            matrix.postRotate(rotationDegrees);
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bm,
                    bm.getWidth(), bm.getHeight(), true);
            bitmap = Bitmap.createBitmap(scaledBitmap, 0, 0,
                    scaledBitmap.getWidth(), scaledBitmap.getHeight(), matrix, true);
        }
        return bitmap;
    }

    private String[] getRequiredPermissions() {
        try {
            PackageInfo info =
                    getPackageManager()
                            .getPackageInfo(getPackageName(), PackageManager.GET_PERMISSIONS);
            String[] ps = info.requestedPermissions;
            if (ps != null && ps.length > 0) {
                return ps;
            } else {
                return new String[0];
            }
        } catch (Exception e) {
            return new String[0];
        }
    }
    private boolean allPermissionsGranted() {
        for (String permission : getRequiredPermissions()) {
            int status= ContextCompat.checkSelfPermission(this,permission);
            if (ContextCompat.checkSelfPermission(this,permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }



    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        switch (requestCode) {
            case REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS:
                Map<String, Integer> perms = new HashMap<String, Integer>();
                boolean allGranted = true;
                for (int i = 0; i < permissions.length; i++) {
                    perms.put(permissions[i], grantResults[i]);
                    if (grantResults[i] != PackageManager.PERMISSION_GRANTED)
                        allGranted = false;
                }
                // Check for ACCESS_FINE_LOCATION
                if (allGranted) {
                    // All Permissions Granted
                    init();
                } else {
                    // Permission Denied
                    Toast.makeText(MainActivity.this, "Some Permission is Denied", Toast.LENGTH_SHORT)
                            .show();
                    finish();
                }
                break;
            default:
                super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }
}