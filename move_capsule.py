# Gerekli kütüphaneleri içe aktarıyoruz
import rclpy
from rclpy.node import Node
# DEĞİŞİKLİK: 'SetEntityState' yerine 'SetEntityPose' import ediyoruz
from ros_gz_interfaces.srv import SetEntityPose 
# DEĞİŞİKLİK: Model adını ve türünü belirtmek için 'Entity' mesajını da import ediyoruz
from ros_gz_interfaces.msg import Entity
from geometry_msgs.msg import Pose, Point, Quaternion

class EntityMover(Node):
    """
    Gazebo'daki bir modeli belirli bir konuma taşımak için bir servis istemcisi oluşturan düğüm.
    """
    def __init__(self):
        # Düğümü 'entity_mover_node' adıyla başlatıyoruz
        super().__init__('entity_mover_node')
        
        # DEĞİŞİKLİK: Servis türünü 'SetEntityPose' olarak güncelliyoruz.
        # Servis adı hala aynı: /world/<dünya_adı>/set_entity_state. ROS 2 bu isimdeki
        # servisin 'SetEntityPose' türünde olduğunu otomatik olarak anlar.
        self.client = self.create_client(SetEntityPose, '/world/corridor/set_entity_state')

        # Servisin aktif hale gelmesini bekliyoruz.
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Servis bulunamadı, bekleniyor...')

    def move_entity(self, name, x, y, z):
        """
        Belirtilen modeli istenen koordinatlara taşımak için servis isteği gönderir.
        """
        # DEĞİŞİKLİK: İstek nesnesini 'SetEntityPose.Request()' olarak oluşturuyoruz.
        request = SetEntityPose.Request()

        # DEĞİŞİKLİK: İstek yapısı biraz farklı. Model adı 'entity' alanına yazılıyor.
        # Hangi modeli hareket ettireceğimizi belirtiyoruz.
        request.entity = Entity()
        request.entity.name = name
        request.entity.type = Entity.MODEL # Varlık türünün bir MODEL olduğunu belirtiyoruz.
        
        # DEĞİŞİKLİK: Modelin yeni pozisyonunu doğrudan 'pose' alanına yazıyoruz.
        request.pose.position.x = x
        request.pose.position.y = y
        request.pose.position.z = z
        
        # Modelin yönelimini (orientation) ayarlıyoruz.
        request.pose.orientation.x = 0.0
        request.pose.orientation.y = 0.0
        request.pose.orientation.z = 0.0
        request.pose.orientation.w = 1.0

        # Servisi asenkron olarak çağırıyoruz ve gelecek (future) nesnesini saklıyoruz.
        self.future = self.client.call_async(request)
        self.get_logger().info(f"'{name}' modeli ({x}, {y}, {z}) konumuna taşınıyor...")

def main(args=None):
    # ROS 2 Python istemcisini başlatıyoruz.
    rclpy.init(args=args)

    # Düğümümüzden bir nesne oluşturuyoruz.
    entity_mover = EntityMover()

    # 'capsule' modelini x=2.0, y=3.0, z=0.5 koordinatlarına taşıyoruz.
    entity_mover.move_entity('capsule', 2.0, 3.0, 0.5)

    # Servis çağrısı tamamlanana kadar düğümü "döndürüyoruz" (spin).
    while rclpy.ok():
        rclpy.spin_once(entity_mover)
        if entity_mover.future.done():
            try:
                # DEĞİŞİKLİK: Yanıtı alırken 'result()' yeterli. Yanıtın içeriği biraz farklı olabilir
                # ama 'success' alanı genellikle bulunur. Bizim kodumuzda bu kısmı değiştirmeye gerek yok.
                response = entity_mover.future.result()
                if response.success:
                    entity_mover.get_logger().info('Model başarıyla taşındı!')
                else:
                    entity_mover.get_logger().error('Model taşınamadı.')
            except Exception as e:
                entity_mover.get_logger().error(f'Servis çağrısı başarısız oldu: {e}')
            break # Döngüden çık

    # Düğümü ve ROS 2'yi kapatıyoruz.
    entity_mover.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
