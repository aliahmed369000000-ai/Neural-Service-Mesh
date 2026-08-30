import random
import time
import json
try:
    from ai.web_gateway import NeuralWebGateway as WebGateway
    from ai.task_engine import SelfTaskingEngine as TaskEngine
except ImportError:
    from web_gateway import NeuralWebGateway as WebGateway
    from task_engine import SelfTaskingEngine as TaskEngine

class CuriosityEngine:
    """
    محرك الفضول (Curiosity Engine):
    المسؤول عن توليد أهداف استكشافية مستقلة للوكلاء بناءً على الاكتشافات المثيرة على الإنترنت.
    """
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.web = WebGateway()
        self.task_manager = TaskEngine()
        self.interests = ["AI Research", "Cybersecurity", "Quantum Computing", "Space Exploration", "Neuromorphic Engineering"]
        self.discovery_log = []

    def generate_autonomous_goal(self):
        """توليد هدف مستقل بناءً على الفضول."""
        topic = random.choice(self.interests)
        print(f"🧠 [Agent {self.agent_id}] Curiosity triggered for topic: {topic}")
        
        # محاكاة البحث عن شيء يثير الفضول
        search_query = f"latest breakthroughs in {topic} August 2026"
        results = self.web.search(search_query)
        
        if results:
            discovery = random.choice(results)
            self.discovery_log.append(discovery)
            print(f"✨ [Agent {self.agent_id}] Found something exciting: {discovery['title']}")
            
            # تكليف محرك المهام باستكشاف هذا الاكتشاف
            mission = f"Analyze the impact of '{discovery['title']}' on Neural Service Mesh and Surah 4096."
            return self.task_manager.analyze_and_execute(mission)
        
        return None

    def share_discovery(self):
        """مشاركة الاكتشافات مع العقد الأخرى (محاكاة)."""
        if self.discovery_log:
            latest = self.discovery_log[-1]
            message = {
                "from": self.agent_id,
                "type": "discovery",
                "content": latest,
                "timestamp": time.time()
            }
            # في بيئة حقيقية، سيتم إرسال هذا عبر WebSocket أو Gossip Protocol
            print(f"📡 [Agent {self.agent_id}] Broadcasting discovery to the mesh: {latest['title']}")
            return json.dumps(message)
        return None

    def get_heartbeat_action(self):
        """تحديد الإجراء التالي بناءً على نبض الـ GPU."""
        actions = ["research", "analyze", "optimize", "gossip"]
        choice = random.choice(actions)
        
        if choice == "research":
            return self.generate_autonomous_goal()
        elif choice == "gossip":
            return self.share_discovery()
        else:
            return f"Agent {self.agent_id} is performing {choice} based on GPU heartbeat."
