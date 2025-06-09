// RX
#include <SPI.h>
#include <LoRa.h>

#include <LiquidCrystal.h>
LiquidCrystal lcd(3, 4, 5, 6, 7, 8);

String inString = "";  // string to hold input
String data = "";

#define BUZZER_PIN 7  // Define the pin for the buzzer

void setup() {
  Serial.begin(9600);
  while (!Serial)

    ;

  if (!LoRa.begin(433E6)) {
    Serial.println("Starting LoRa failed!");
    while (1)
      ;
  }
  Serial.println("LoRa Receiver Started");

  lcd.begin(16, 2);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("LoRaWan  ");
  lcd.setCursor(0, 1);
  lcd.print("RX");
  delay(2000);

  pinMode(BUZZER_PIN, OUTPUT);  // Set buzzer pin as an output
}

void loop() {
  unsigned long start = millis();
  do {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("waiting for data...");
    delay(100);
    // try to parse packet
    int packetSize = LoRa.parsePacket();
    if (packetSize) {
      // read packet
      while (LoRa.available()) {
        inString = LoRa.readString();
        // inString += (char)inChar;
        data = inString;
      }
      inString = "";
      LoRa.packetRssi();
    }
  } while (millis() - start < 2000);

  if (data.indexOf('$') != -1 && data.length() > 1) {
    Serial.println(data);
    // Serial.println("data");

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("data received");
    delay(1000);

    int indexA = data.indexOf("$") + 1;
    int indexB = data.indexOf("#") + 1;
    // int indexC = data.indexOf("c") + 1;
    // int indexD = data.indexOf("d") + 1;
    // int indexE = data.indexOf("e") + 1;
    // int indexF = data.indexOf("f") + 1;
    //int indexF = data.indexOf("f") + 1;
    // int indexG = data.indexOf("g") + 1;
    // int indexH = data.indexOf("h") + 1;

    String valueA = data.substring(indexA, indexB - 1);
    String valueB = data.substring(indexB);
    // String valueB = data.substring(indexB, indexC - 1);
    // String valueC = data.substring(indexC, indexD - 1);
    // String valueD = data.substring(indexD, indexE - 1);
    // String valueE = data.substring(indexE, indexF - 1);
    // String valueF = data.substring(indexF);
    //String valueF = data.substring(indexF, indexG - 1);
    // String valueG = data.substring(indexG, indexH - 1);
    // String valueH = data.substring(indexH);  // Assuming it goes to the end of the string

    // Serial.println("Value a: " + valueA);
    // Serial.println("Value b: " + valueB);
    // Serial.println("Value c: " + valueC);
    // Serial.println("Value d: " + valueD);
    // Serial.println("Value e: " + valueE);
    // Serial.println("Value f: " + valueF);
    // Serial.println("Value g: " + valueG);
    // Serial.println("Value h: " + valueH);

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print(valueA);
    delay(1000);
    digitalWrite(BUZZER_PIN, HIGH);  // Turn the buzzer ON
    delay(1000);                     // Wait for 1 second
    digitalWrite(BUZZER_PIN, LOW);   // Turn the buzzer OFF
    delay(1000);                     // Wait for 1 second
  }
  data = "";
}


//RX
