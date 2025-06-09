#include <SPI.h>
#include <LoRa.h>

#include <LiquidCrystal.h>
LiquidCrystal lcd(A2, 4, 5, 6, 7, 8);

String data = "";

#define BUZZER_PIN 7  // Define the pin for the buzzer

void setup() {
  Serial.begin(9600);  // initialize the serial monitor

  while (!Serial)

    ;
  if (!LoRa.begin(433E6)) {
    Serial.println("Starting LoRa failed!");
    while (1)
      ;
  }
  Serial.println("LoRa sender Started");
  lcd.begin(16, 2);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("LoRaWan  ");
  lcd.setCursor(0, 1);
  lcd.print("TX");
  delay(2000);
  pinMode(BUZZER_PIN, OUTPUT);  // Set buzzer pin as an output
}

void loop() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("waiting for data");
  lcd.setCursor(0, 1);
  lcd.print("pi");
  delay(1000);
  data = Serial.readStringUntil("\n");

  if (data.indexOf('$') != -1 && data.length() > 1) {
    Serial.println(data);
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print(data);
    delay(2000);
    loratx();
    digitalWrite(BUZZER_PIN, HIGH);  // Turn the buzzer ON
    delay(1000);                     // Wait for 1 second
    digitalWrite(BUZZER_PIN, LOW);   // Turn the buzzer OFF
    delay(1000);
  }
}

void loratx() {
  unsigned long start = millis();
  do {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("data sending...");
    delay(100);
    LoRa.beginPacket();
    LoRa.print(data);
    Serial.println(data);
    LoRa.endPacket();
  } while (millis() - start < 2000);
  lcd.clear();
  lcd.setCursor(0, 1);
  lcd.print("data send.....");
  // Serial.println("data");
  delay(2000);
}
