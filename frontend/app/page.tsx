"use client";

import { useState } from "react";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { toast } from "sonner";
import { Gamepad2, Loader2, Download, Play, Square } from "lucide-react";

interface Action {
  id: "w" | "a" | "s" | "d";
  speed: number;
  frames: number;
}

export default function Home() {
  const [imageUrl, setImageUrl] = useState("https://v3.fal.media/files/lion/tr5WjeDTlYUohGnwA0h03.jpeg");
  const [prompt, setPrompt] = useState("A charming medieval village with cobblestone streets, timber-framed buildings, and a central fountain. The village is bustling with villagers going about their daily activities.");
  const [actions, setActions] = useState<Action[]>([
    { id: "w", speed: 0.2, frames: 33 }
  ]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoMetadata, setVideoMetadata] = useState<any>(null);

  const addAction = () => {
    setActions([...actions, { id: "w", speed: 0.2, frames: 33 }]);
  };

  const removeAction = (index: number) => {
    setActions(actions.filter((_, i) => i !== index));
  };

  const updateAction = (index: number, field: keyof Action, value: any) => {
    const newActions = [...actions];
    newActions[index] = { ...newActions[index], [field]: value };
    setActions(newActions);
  };

  const generateVideo = async () => {
    setIsGenerating(true);
    setVideoUrl(null);
    setVideoMetadata(null);

    try {
      const response = await axios.post(
        "https://fal.run/Remade-AI/81f9edaa-90ce-4c87-b98f-aabeb520566c/generate",
        {
          image_url: imageUrl,
          prompt: prompt,
          negative_prompt: "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.",
          actions: actions,
          size: [704, 1216],
          infer_steps: 8,
          guidance_scale: 1.0,
          seed: 250160,
          use_fp8: false,
          cpu_offload: false
        },
        {
          headers: {
            "Content-Type": "application/json"
          },
          timeout: 600000 // 10 minutes
        }
      );

      if (response.data.video?.url) {
        setVideoUrl(response.data.video.url);
        setVideoMetadata({
          width: response.data.width,
          height: response.data.height,
          totalFrames: response.data.total_frames
        });
        toast.success("Your video has been generated successfully.");
      }
    } catch (error: any) {
      console.error("Error generating video:", error);
      toast.error(error.response?.data?.message || "Failed to generate video. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  const getTotalFrames = () => {
    return actions.reduce((total, action) => total + action.frames, 0);
  };

  return (
    <div className="min-h-screen bg-[#F5F1E8]">
      {/* Header with FAL-inspired branding */}
      <header className="border-b border-black/10 bg-white/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center gap-4">
            {/* Pixelated Game Controller Icon */}
            <div className="relative">
              <Gamepad2 className="h-10 w-10" style={{ imageRendering: "pixelated" }} />
              <div className="absolute inset-0 bg-black/5 mix-blend-multiply" style={{
                clipPath: "polygon(20% 0%, 40% 0%, 40% 20%, 60% 20%, 60% 0%, 80% 0%, 80% 20%, 100% 20%, 100% 40%, 80% 40%, 80% 60%, 100% 60%, 100% 80%, 80% 80%, 80% 100%, 60% 100%, 60% 80%, 40% 80%, 40% 100%, 20% 100%, 20% 80%, 0% 80%, 0% 60%, 20% 60%, 20% 40%, 0% 40%, 0% 20%, 20% 20%)"
              }} />
            </div>
            <div>
              <h1 className="text-4xl font-bold tracking-tight">GameCraft</h1>
              <p className="text-sm text-gray-600">Interactive Video is Next</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="grid gap-8 lg:grid-cols-2">
          {/* Input Section */}
          <div className="space-y-6">
            <Card className="p-6 bg-white/80 backdrop-blur-sm border-black/10">
              <h2 className="text-2xl font-semibold mb-4">Create Your Game World</h2>
              
              {/* Image URL Input */}
              <div className="space-y-2 mb-4">
                <label className="text-sm font-medium">Reference Image URL</label>
                <Input
                  value={imageUrl}
                  onChange={(e) => setImageUrl(e.target.value)}
                  placeholder="https://example.com/image.jpg"
                  className="font-mono text-sm"
                />
              </div>

              {/* Prompt Input */}
              <div className="space-y-2 mb-4">
                <label className="text-sm font-medium">Scene Description</label>
                <Textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Describe your game world..."
                  className="min-h-[100px]"
                />
              </div>

              {/* Actions Builder */}
              <div className="space-y-2 mb-4">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Camera Actions</label>
                  <Button
                    onClick={addAction}
                    size="sm"
                    variant="outline"
                    className="text-xs"
                  >
                    Add Action
                  </Button>
                </div>
                
                <div className="space-y-2 max-h-[300px] overflow-y-auto">
                  {actions.map((action, index) => (
                    <div key={index} className="flex gap-2 p-3 bg-gray-50 rounded-lg">
                      <Select
                        value={action.id}
                        onValueChange={(value) => updateAction(index, "id", value)}
                      >
                        <SelectTrigger className="w-24">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="w">↑ (W)</SelectItem>
                          <SelectItem value="s">↓ (S)</SelectItem>
                          <SelectItem value="a">← (A)</SelectItem>
                          <SelectItem value="d">→ (D)</SelectItem>
                        </SelectContent>
                      </Select>
                      
                      <Input
                        type="number"
                        value={action.speed}
                        onChange={(e) => updateAction(index, "speed", parseFloat(e.target.value))}
                        placeholder="Speed"
                        className="w-20"
                        step="0.1"
                        min="0.1"
                        max="1"
                      />
                      
                      <Input
                        type="number"
                        value={action.frames}
                        onChange={(e) => updateAction(index, "frames", parseInt(e.target.value))}
                        placeholder="Frames"
                        className="w-20"
                        min="1"
                        max="100"
                      />
                      
                      <Button
                        onClick={() => removeAction(index)}
                        size="sm"
                        variant="ghost"
                        className="px-2"
                      >
                        <Square className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
                
                <div className="flex justify-between text-sm text-gray-600">
                  <span>{actions.length} actions</span>
                  <span>{getTotalFrames()} total frames</span>
                </div>
              </div>

              {/* Generate Button */}
              <Button
                onClick={generateVideo}
                disabled={isGenerating || !imageUrl || !prompt || actions.length === 0}
                className="w-full h-12 text-lg font-semibold"
                style={{
                  background: isGenerating ? "#6B7280" : "#000000",
                  imageRendering: "pixelated"
                }}
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Generating World...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-5 w-5" />
                    Generate Video
                  </>
                )}
              </Button>
            </Card>
          </div>

          {/* Output Section */}
          <div className="space-y-6">
            <Card className="p-6 bg-white/80 backdrop-blur-sm border-black/10">
              <h2 className="text-2xl font-semibold mb-4">Your Generated World</h2>
              
              {isGenerating ? (
                <div className="space-y-4">
                  <Skeleton className="aspect-[9/16] w-full" />
                  <div className="space-y-2">
                    <Skeleton className="h-4 w-3/4" />
                    <Skeleton className="h-4 w-1/2" />
                  </div>
                </div>
              ) : videoUrl ? (
                <div className="space-y-4">
                  <div className="relative aspect-[9/16] bg-black rounded-lg overflow-hidden">
                    <video
                      src={videoUrl}
                      controls
                      className="w-full h-full object-contain"
                      autoPlay
                      loop
                    />
                  </div>
                  
                  {videoMetadata && (
                    <div className="flex gap-2 flex-wrap">
                      <Badge variant="secondary">
                        {videoMetadata.width}×{videoMetadata.height}
                      </Badge>
                      <Badge variant="secondary">
                        {videoMetadata.totalFrames} frames
                      </Badge>
                    </div>
                  )}
                  
                  <Button
                    asChild
                    variant="outline"
                    className="w-full"
                  >
                    <a href={videoUrl} download="gamecraft-video.mp4">
                      <Download className="mr-2 h-4 w-4" />
                      Download Video
                    </a>
                  </Button>
                </div>
              ) : (
                <div className="aspect-[9/16] bg-gray-100 rounded-lg flex items-center justify-center">
                  <p className="text-gray-500 text-center px-4">
                    Your generated video will appear here
                  </p>
                </div>
              )}
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}