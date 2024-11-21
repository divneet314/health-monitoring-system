#include <Wire.h>
#include <PulseSensorPlayground.h>
#include <U8g2lib.h>
#include <WiFi.h>
#include <HTTPClient.h>

// Replace with your network credentials
const char* ssid = "Security";
const char* password = "";


// Pin definitions
#define PULSE_SENSOR_PIN 4
#define LED_PIN 2  // On-board LED pin (usually pin 2 on ESP32)

PulseSensorPlayground pulseSensor;  // Create a PulseSensor object

// Initialize u8g2 library for SSD1306 (128x64) using I2C
U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, /* reset=*/ U8X8_PIN_NONE, /* clock=*/ SCL, /* data=*/ SDA);

// Variable to store heart rate
int heartRate = 0;

// Buffer for heart rate history (stores last 128 values)
#define MAX_HISTORY 128
int pulseDataHistory[MAX_HISTORY];
int historyIndex = 0;

void setup() {
  // Start Serial Monitor
  Serial.begin(9600);
  
  // Initialize the Pulse Sensor
  pulseSensor.analogInput(PULSE_SENSOR_PIN);
  pulseSensor.blinkOnPulse(LED_PIN);  // Blink the onboard LED when a heartbeat is detected
  pulseSensor.setThreshold(550);  // Set threshold for detecting a beat

  if (pulseSensor.begin() == false) {
    Serial.println("Pulse sensor not found!");
    while (true);
  }

  // Initialize OLED display
  u8g2.begin();
  u8g2.setFont(u8g2_font_ncenB08_tr);  // Set font for text

  // Initialize pulse data history buffer
  for (int i = 0; i < MAX_HISTORY; i++) {
    pulseDataHistory[i] = 0;  // Fill the history buffer with 0 initially
  }

  Serial.println("Pulse sensor and OLED ready.");
  WiFi.begin(ssid, password);

  Connect to Wi-Fi
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("\nConnected to Wi-Fi");
}

void loop() {
  // Capture the pulse sensor signal (raw data)
  int signal = analogRead(PULSE_SENSOR_PIN)/3; 
  // if(signal > Threshold){                          
  //    digitalWrite(2,HIGH);
  //  } else {
  //    digitalWrite(2,LOW);                
  //  }
  // Check if a new heartbeat is detected
  if (pulseSensor.sawStartOfBeat()) {
    int bpm = pulseSensor.getBeatsPerMinute();
    Serial.println("♥ A Heartbeat Happened!");
    Serial.print("BPM: ");
    Serial.println(bpm);
    

    // Update heart rate variable
    heartRate = bpm;
  }

  // Update pulse data history buffer
  pulseDataHistory[historyIndex] = signal;
  historyIndex = (historyIndex + 1) % MAX_HISTORY; 

  // Display heart rate on OLED screen
  u8g2.clearBuffer();  
  u8g2.setCursor(0, 12);
  u8g2.print("Heart Rate: ");
  u8g2.print(heartRate);

  // Draw the heartbeat wave below the current BPM value
  drawHeartbeatWave();

  u8g2.sendBuffer();  // Send buffer to OLED display
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;

    // Replace with your server URL
    String url = "https://api.thingspeak.com/update?api_key=Z0A3OMDJY9C9XATK&field2=82";
    http.begin(url);

    int httpResponseCode = http.GET();

    if (httpResponseCode > 0) {
      String payload = http.getString();
      Serial.println("HTTP Response code: " + String(httpResponseCode));
      Serial.println("Response payload: " + payload);
    } else {
      Serial.println("Error on HTTP request");
    }

    http.end();
  } else {
    Serial.println("Wi-Fi not connected");
  }
  delay(200);  // Small delay for stability
}

// Function to draw the heartbeat wave
void drawHeartbeatWave() {
  int graphHeight = 30;  // Height of the waveform graph
  int graphWidth = 128;  // Width of the graph (full display width)

  // Draw the wave axis
  u8g2.setDrawColor(1);  // Set color for drawing
  u8g2.drawLine(0, 48, graphWidth, 48);  // Draw the X-axis at 48px position

  // Plot the heartbeat wave based on pulse data
  for (int i = 0; i < MAX_HISTORY - 1; i++) {
    int x1 = i * (graphWidth / MAX_HISTORY);  // X position for the first point
    int x2 = (i + 1) * (graphWidth / MAX_HISTORY);  // X position for the second point
    int y1 = map(pulseDataHistory[i], 0, 1023, 0, graphHeight);  // Y position for the first point (mapped from pulse sensor range)
    int y2 = map(pulseDataHistory[i + 1], 0, 1023, 0, graphHeight);  // Y position for the second point

    u8g2.drawLine(x1, 48 - y1, x2, 48 - y2);  // Draw the line between two points to create the wave
  }
}
