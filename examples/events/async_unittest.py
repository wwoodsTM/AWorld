import asyncio
import signal
import time
import random
from functools import partial

from aworld.events.async_event import AsyncEvent
from aworld.events.async_pub_event import publish_async_event
from aworld.events.async_sub_event import sub_async_event

APPLICATION_GROUP = "APP"
SECURITY_GROUP = "SECURITY"
MONITORING_GROUP = "MONITORING"

# Define async event classes inheriting from AsyncEvent
class AsyncAppStartupEvent(AsyncEvent):
    def __init__(self, app_name, **kwargs):
        super().__init__(
            event_code="APP_STARTUP",
            event_group=APPLICATION_GROUP,
            **kwargs
        )
        self.app_name = app_name


class AsyncAppShutdownEvent(AsyncEvent):
    def __init__(self, app_name, **kwargs):
        super().__init__(
            event_code="APP_SHUTDOWN",
            event_group=APPLICATION_GROUP,
            **kwargs
        )
        self.app_name = app_name


class AsyncSecurityAlertEvent(AsyncEvent):
    def __init__(self, threat_name, threat_level=1, **kwargs):
        super().__init__(
            event_code="SECURITY_ALERT",
            event_group=SECURITY_GROUP,
            **kwargs
        )
        self.threat_name = threat_name
        self.threat_level = threat_level


class AsyncSystemMetricEvent(AsyncEvent):
    def __init__(self, metric_name, metric_value, **kwargs):
        super().__init__(
            event_code="SYSTEM_METRIC",
            event_group=MONITORING_GROUP,
            **kwargs
        )
        self.metric_name = metric_name
        self.metric_value = metric_value


# Global variable to control program execution
running = True


# Async event simulator
async def async_event_simulator():
    """Async event simulator that periodically generates various types of events"""
    i = 0
    while running:
        event_type = i % 3  # Cycle through three different event types
        
        if event_type == 0:
            # Application events
            if random.choice([True, False]):
                # App startup event
                event = AsyncAppStartupEvent(
                    app_name=f"AppService_{random.randint(1, 5)}"
                )
            else:
                # App shutdown event
                event = AsyncAppShutdownEvent(
                    app_name=f"AppService_{random.randint(1, 5)}"
                )
        elif event_type == 1:
            # Security events
            event = AsyncSecurityAlertEvent(
                threat_name=f"SecurityThreat_{random.randint(1, 3)}",
                threat_level=random.randint(1, 5)
            )
        else:
            # Monitoring metric events
            metrics = ["CPU", "Memory", "Disk", "Network", "Temperature"]
            event = AsyncSystemMetricEvent(
                metric_name=random.choice(metrics),
                metric_value=random.uniform(0, 100)
            )
            
        # Async publish event
        await publish_async_event(event)
        
        i += 1
        # Random delay between 0.5-1.5 seconds
        delay = random.uniform(0.5, 1.5)
        await asyncio.sleep(delay)


# Define async event handlers with decorators
@sub_async_event(event_group=APPLICATION_GROUP)
async def async_app_event_monitor(event):
    """Async handler for all application events"""
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{current_time}] Application Event: {event.event_code} - App: {event.app_name}")
    # Simulate async processing time
    await asyncio.sleep(0.2)
    print(f"✓ Application event {event.event_id} processing completed")


@sub_async_event(event_group=SECURITY_GROUP)
async def async_security_event_monitor(event):
    """Async handler for all security events"""
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{current_time}] ⚠️ Security Alert: {event.threat_name} (Level: {event.threat_level})")
    
    # Dynamically adjust processing time based on threat level
    processing_time = 0.1 * event.threat_level
    await asyncio.sleep(processing_time)
    
    print(f"✓ Security event {event.event_id} processing completed")


@sub_async_event(event_group=MONITORING_GROUP)
async def async_metric_monitor(event):
    """Async handler for system monitoring metric events"""
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{current_time}] 📊 System Metric: {event.metric_name} = {event.metric_value:.2f}")
    
    # Simulate metric processing and analysis
    await asyncio.sleep(0.1)
    
    # Special handling for abnormal values
    if event.metric_value > 80:
        print(f"⚠️ {event.metric_name} metric exceeds threshold: {event.metric_value:.2f}")
        # Simulate additional processing time for high values
        await asyncio.sleep(0.2)
    
    print(f"✓ Metric event {event.event_id} processing completed")


# Wildcard subscription example - logs all events
@sub_async_event()
async def async_event_logger(event):
    """Async logging handler for all events"""
    # Very short processing time needed
    await asyncio.sleep(0.05)
    # Don't print detailed information to avoid excessive logs
    # print(f"Log: Recorded event {event.event_id} [{event.event_group}:{event.event_code}]")


# Combined condition subscription example - high-level security threats
@sub_async_event(event_group=SECURITY_GROUP)
async def async_high_threat_handler(event):
    """Special handler for high-level security threats"""
    if hasattr(event, 'threat_level') and event.threat_level >= 4:
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{current_time}] 🔴 High-level Threat Alert: {event.threat_name} (Level: {event.threat_level})")
        
        # High-level threats need more processing time
        await asyncio.sleep(0.5)
        
        print(f"✓ High-level threat {event.event_id} notified to security team")


# Signal handler function
def signal_handler(sig, frame):
    """Handle termination signals"""
    global running
    print("\nReceived termination signal, gracefully shutting down...")
    running = False


async def main():
    """Async main function"""
    print("=== Async Event System Demo ===")
    print("This demo showcases the usage of the async event system, including:")
    print(f"  - {APPLICATION_GROUP}: Application events (startup/shutdown)")
    print(f"  - {SECURITY_GROUP}: Security events (threat alerts)")
    print(f"  - {MONITORING_GROUP}: Monitoring events (system metrics)")
    print("Each event type has its own async handler.")
    print("\nPress Ctrl+C to stop the service...\n")
    
    try:
        # Start event simulator
        simulator_task = asyncio.create_task(async_event_simulator())
        
        # Keep the program running until termination signal is received
        while running:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        print("Main task canceled")
    except Exception as e:
        print(f"Async event simulator error: {e}")
    finally:
        print("\nService stopped.")


if __name__ == "__main__":
    # Register signal handlers, referencing unittest.py approach
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(main())
    finally:
        print("Program exited.") 