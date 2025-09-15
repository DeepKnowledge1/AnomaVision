import socket
import cv2
import os
import time
import struct
import sys

def send_images(host="0.0.0.0", port=5000, image_folder=None, use_webcam=False):
    """
    TCP server to send images to clients.

    Args:
        host (str): Host/IP to bind the server.
        port (int): Port to listen on.
        image_folder (str): Folder containing images to send.
        use_webcam (bool): If True, capture frames from webcam instead of folder.
    """
    # Setup TCP server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"[SERVER] Listening on {host}:{port}")

    while True:
        try:
            conn, addr = server_socket.accept()
            print(f"[SERVER] Client connected: {addr}")

            if use_webcam:
                cap = cv2.VideoCapture(0)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if not send_frame(conn, frame):
                        break  # client disconnected
                    time.sleep(0.1)  # ~10 FPS
                cap.release()
            else:
                images = sorted(os.listdir(image_folder))
                for img_name in images:
                    img_path = os.path.join(image_folder, img_name)
                    frame = cv2.imread(img_path)
                    if frame is None:
                        continue
                    if not send_frame(conn, frame):
                        break
                    time.sleep(0.5)  # Send one every 0.5s

        except KeyboardInterrupt:
            print("\n[SERVER] Shutting down...")
            break
        except Exception as e:
            print(f"[SERVER] Error: {e}", file=sys.stderr)
        finally:
            try:
                conn.close()
            except:
                pass
            print("[SERVER] Waiting for next client...")

    server_socket.close()

def send_frame(conn, frame):
    """Encode frame as JPEG and send with length prefix"""
    try:
        _, encoded = cv2.imencode(".jpg", frame)
        data = encoded.tobytes()
        size = len(data)

        # Send 4-byte size header + data
        conn.sendall(struct.pack(">I", size) + data)
        print(f"[SERVER] Sent frame ({size} bytes)")
        return True
    except (BrokenPipeError, ConnectionResetError):
        print("[SERVER] Client disconnected")
        return False
    except Exception as e:
        print(f"[SERVER] Send error: {e}")
        return False

if __name__ == "__main__":
    # Example: send webcam frames
    send_images(port=5000, use_webcam=True)
    # Or send from a folder:
    # send_images(port=5000, image_folder="images")
