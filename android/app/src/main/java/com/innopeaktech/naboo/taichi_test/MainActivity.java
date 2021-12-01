package com.innopeaktech.naboo.taichi_test;

import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    private VKSurfaceView vkSurfaceView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        vkSurfaceView = new VKSurfaceView(this);
        setContentView(vkSurfaceView);
    }
}
