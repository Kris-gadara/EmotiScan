package com.asav.facialprocessing;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.util.Base64;
import android.util.Log;
import android.widget.Toast;

import org.json.*;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

import okhttp3.*;

import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSession;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;

public class ServerProcessor {
    private static final String TAG = "ServerProcessor";
    private static final MediaType JSON = MediaType.parse("application/json; charset=utf-8");
    private String URL_BASE = "";//https://beta.saluteai.sberdevices.ru/v1/chat/completions";
    public boolean isEnabled = true;
    private OkHttpClient client ;
    private final Activity context;
    private String token;
    private String serverAddress="";

    public ServerProcessor(Activity context, String serverAddress) {
        this.context = context;
        this.serverAddress=serverAddress;
        URL_BASE = "http://".concat(serverAddress).concat(":5050");

        try {
            client = new OkHttpClient.Builder()
                    .connectTimeout(10, TimeUnit.SECONDS)
                    .writeTimeout(1, TimeUnit.MINUTES)
                    .readTimeout(1, TimeUnit.MINUTES)
                    .build();

        }catch(Exception ex){
            Log.e(TAG, "Failed to connect to server", ex);
            isEnabled=false;
        }
    }
    public void networkStateChanged(boolean networkState) {
        isEnabled = networkState;
    }
    public String getServerAddress(){
        return serverAddress;
    }

    public Map<String,String> insideout(String question, String userEmotion) {
        if(!isEnabled) {
            Log.e(TAG, "Network is unavailable");
            return null;
        }
        try {
            Request request = new Request.Builder()
                    .url(URL_BASE.concat("/insideout"))
                    //.url(URL_BASE.concat("/single"))
                    .post(RequestBody.create(JSON, String.format("{\"question\": \"%s\", \"userEmotion\": \"%s\"}", question, userEmotion)))
                    .build();
            Response response = client.newCall(request).execute();
            assert response.body() != null;
            String resp=response.body().string();
            //Log.d(TAG,resp);
            JSONObject jsonResp=new JSONObject(resp);
            return processResult(jsonResp);
        } catch(Exception ex){
            Log.e(TAG, "Could not process request on the server", ex);
            isEnabled = false;
            return null;
        }
    }
    public Map<String,String> single_call(String question, String userEmotion) {
        if(!isEnabled) {
            Log.e(TAG, "Network is unavailable");
            return null;
        }
        try {
            Request request = new Request.Builder()
                    .url(URL_BASE.concat("/single"))
                    .post(RequestBody.create(JSON, String.format("{\"question\": \"%s\", \"userEmotion\": \"%s\"}", question, userEmotion)))
                    .build();
            Response response = client.newCall(request).execute();
            assert response.body() != null;
            String resp=response.body().string();
            //Log.d(TAG,resp);
            JSONObject jsonResp=new JSONObject(resp);
            return processResult(jsonResp);
        } catch(Exception ex){
            Log.e(TAG, "Could not process request on the server", ex);
            isEnabled = false;
            return null;
        }
    }
    private Map<String,String> processResult(JSONObject result)  throws  JSONException{
        //Log.e(TAG, "Result:"+result);
        Map<String,String> res=new LinkedHashMap<>();
        for (Iterator<String> it = result.keys(); it.hasNext(); ) {
            String k = it.next();
            res.put(k,(String)result.get(k));
        }
        return res;
    }
}
