from huggingface_hub import HfApi

api = HfApi()
try:
    user_info = api.whoami()
    print(f"✅ 登录成功！用户名为: {user_info['name']}")
    
    # 测试是否有权访问具体的 PaliGemma 模型
    model_id = "google/paligemma2-3b-pt-224"
    api.model_info(model_id)
    print(f"✅ 权限确认：你可以访问 {model_id}")
    
except Exception as e:
    print(f"❌ 验证失败: {e}")