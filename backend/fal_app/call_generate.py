import requests
import json
import time

def call_generate_endpoint(image_url):
    """
    Call the /generate endpoint with the provided image URL
    """
    
    # The endpoint URL - you'll need to replace this with your actual FAL endpoint
    # This will be something like: https://your-app-name.gateway.fal.run/generate
    endpoint_url = "https://fal.run/Remade-AI/44182342-81e7-40fc-9b7e-70c3f0d183a6/generate"
    image_url = "https://v3.fal.media/files/lion/tr5WjeDTlYUohGnwA0h03.jpeg"
    # Request payload based on your schema
    payload = {
        "image_url": image_url,
        "prompt": "A charming medieval village with cobblestone streets, timber-framed buildings, and a central fountain. The village is bustling with villagers going about their daily activities.",
        "negative_prompt": "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.",
        "actions": [
            {
                "id": "w",
                "speed": 0.2,
                "frames": 33
            },
            {
                "id": "w",
                "speed": 0.2,
                "frames": 33
            },
            {
                "id": "d",
                "speed": 0.2,
                "frames": 33
            },
            {
                "id": "d",
                "speed": 0.2,
                "frames": 33
            }
        ],
        "size": [704, 1216],
        "infer_steps": 8,
        "guidance_scale": 1.0,
        "seed": 250160,
        "use_fp8": False,
        "cpu_offload": False
    }
    
    # Headers
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Calling endpoint: {endpoint_url}")
        print(f"Image URL: {image_url}")
        print("Sending request...")
        
        # Make the request
        response = requests.post(
            endpoint_url,
            json=payload,
            headers=headers,
            timeout=300  # 5 minute timeout since video generation can take time
        )
        
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success! Video generated.")
            print(f"Video URL: {result.get('video', {}).get('url', 'N/A')}")
            print(f"Height: {result.get('height', 'N/A')}")
            print(f"Width: {result.get('width', 'N/A')}")
            print(f"Total frames: {result.get('total_frames', 'N/A')}")
            print(f"Log: {result.get('log', 'N/A')[:500]}...")  # First 500 chars of log
            
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out. Video generation might still be in progress.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON response: {e}")
        print(f"Response text: {response.text}")
        return None

def main():
    # Replace this with your actual image URL
    image_url = "https://v3.fal.media/files/lion/tr5WjeDTlYUohGnwA0h03.jpeg"
    
    # Call the endpoint
    result = call_generate_endpoint(image_url)
    
    if result:
        print("\n🎉 Video generation completed successfully!")
    else:
        print("\n💥 Video generation failed.")

if __name__ == "__main__":
    main()