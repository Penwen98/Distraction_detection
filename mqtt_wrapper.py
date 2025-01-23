import paho.mqtt.client as mqtt
import json

class MQTTWrapper:
    def __init__(self, host, port, topic):
        self.host = host
        self.port = port
        self.topic = topic
        self.client = mqtt.Client()
        self.client.connect(self.host, self.port)
    
    @staticmethod
    def fromJsonConfig(filename):
        with open(filename) as file:
            config = json.load(file)
        return MQTTWrapper(config["host"], config["port"], config["topic"])

    def publish(self, message):
        self.client.publish(self.topic, message)