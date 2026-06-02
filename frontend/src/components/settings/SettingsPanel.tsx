"use client";

import { Settings } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";

interface SettingsPanelProps {
  settings: Settings;
  onSettingsChange: (settings: Partial<Settings>) => void;
}

export function SettingsPanel({
  settings,
  onSettingsChange,
}: SettingsPanelProps) {
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-lg">⚙️ 设置</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Top-K Slider */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="text-sm font-medium">检索数量 (Top-K)</label>
            <span className="text-sm text-muted-foreground">{settings.top_k}</span>
          </div>
          <Slider
            value={settings.top_k}
            onValueChange={(value) => onSettingsChange({ top_k: Array.isArray(value) ? value[0] : value })}
            min={1}
            max={20}
            step={1}
          />
        </div>

        <Separator />

        {/* Toggle Settings */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label htmlFor="use_memory" className="text-sm font-medium cursor-pointer">
                💬 上下文记忆
              </label>
              <p className="text-xs text-muted-foreground">记住多轮对话内容</p>
            </div>
            <Switch
              id="use_memory"
              checked={settings.use_memory}
              onCheckedChange={(checked) =>
                onSettingsChange({ use_memory: checked })
              }
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label htmlFor="use_stream" className="text-sm font-medium cursor-pointer">
                ⚡ 流式输出
              </label>
              <p className="text-xs text-muted-foreground">逐字流式返回答案</p>
            </div>
            <Switch
              id="use_stream"
              checked={settings.use_stream}
              onCheckedChange={(checked) =>
                onSettingsChange({ use_stream: checked })
              }
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label htmlFor="use_sparse" className="text-sm font-medium cursor-pointer">
                混合检索
              </label>
              <p className="text-xs text-muted-foreground">稠密 + 稀疏 (BM25) 混合检索</p>
            </div>
            <Switch
              id="use_sparse"
              checked={settings.use_sparse}
              onCheckedChange={(checked) =>
                onSettingsChange({ use_sparse: checked })
              }
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label htmlFor="use_reranker" className="text-sm font-medium cursor-pointer">
                🎯 重排序
              </label>
              <p className="text-xs text-muted-foreground">BGE reranker 精排</p>
            </div>
            <Switch
              id="use_reranker"
              checked={settings.use_reranker}
              onCheckedChange={(checked) =>
                onSettingsChange({ use_reranker: checked })
              }
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label htmlFor="allow_model_fallback" className="text-sm font-medium cursor-pointer">
                允许模型常识补充
              </label>
              <p className="text-xs text-muted-foreground">关闭后仅基于可引用上下文回答</p>
            </div>
            <Switch
              id="allow_model_fallback"
              checked={settings.allow_model_fallback}
              onCheckedChange={(checked) =>
                onSettingsChange({ allow_model_fallback: checked })
              }
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label htmlFor="enable_think" className="text-sm font-medium cursor-pointer">
                🧠 深度思考
              </label>
              <p className="text-xs text-muted-foreground">启用 R1 推理模式</p>
            </div>
            <Switch
              id="enable_think"
              checked={settings.enable_think}
              onCheckedChange={(checked) =>
                onSettingsChange({ enable_think: checked })
              }
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
