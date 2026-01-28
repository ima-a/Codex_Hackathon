# 🔗 Connecting RL Model to Frontend

## Quick Integration Guide

### 1. Install API Dependencies
```bash
cd d:\Codex Hackathon\Codex_Hackathon\inventory-hmarl
pip install flask flask-cors
```

### 2. Start Flask API Server  
```bash
# Terminal 1 - Backend API (Python)
cd d:\Codex Hackathon\Codex_Hackathon\inventory-hmarl
python api/recommendation_service.py
```

**Expected output:**
```
🚀 Starting RL-Powered Inventory API Server
====================================================================
Loading RL model from: ../checkpoints/ppo_store_agents_gym.pt
✅ Model loaded successfully!
  
📡 API Endpoints:
   GET  /api/health              - Health check
   GET  /api/recommend           - Get RL recommendation  
   GET  /api/inventory/status    - Get inventory status

🌐 Server starting on http://localhost:5000
====================================================================
```

### 3. Frontend Already Running
```bash
# Terminal 2 - Frontend (React) - ALREADY RUNNING ✅
cd d:\Codex Hackathon\Codex_Hackathon\inventory-hmarl\frontend
npm run dev
# Running on http://localhost:3000
```

### 4. Test the Connection

**Option A: Browser Test**
1. Open http://localhost:3000/inventory-manager
2. Change store/SKU/scenario
3. Look for 🤖 emoji in recommendation reason = **RL MODEL IS WORKING!**
4. Look for 📊 emoji = fallback calculation

**Option B: Direct API Test**
```bash
curl "http://localhost:5000/api/health"
curl "http://localhost:5000/api/recommend?store=store_1&sku=SKU_001&scenario=spike"
```

### 5. Frontend Integration Status

The frontend file `InventoryManager.tsx` needs one small update to call the API.

Add this at the top of the component (after line 20):

```typescript
// API Configuration
const API_BASE_URL = 'http://localhost:5000';
const USE_RL_API = true; // Toggle to switch between API and fallback
```

Then modify the useEffect around line 102 to:

```typescript
useEffect(() => {
  const fetchRecommendation = async () => {
    if (USE_RL_API) {
      try {
        const response = await fetch(
          `${API_BASE_URL}/api/recommend?store=${selectedStore}&sku=${selectedSKU}&scenario=${scenario}`
        );
        const data = await response.json();
        setRecommendation({
          orderQuantity: data.orderQuantity,
          reason: data.reason,
          expectedServiceLevel: data.expectedServiceLevel,
          stockoutRisk: data.stockoutRisk,
          estimatedCost: data.estimatedCost,
        });
        return;
      } catch (error) {
        console.warn('API unavailable, using fallback');
      }
    }
    
    // Fallback calculation (existing code)
    // ... keep existing recommendation logic ...
  };
  
  fetchRecommendation();
}, [scenario, selectedStore, selectedSKU, currentStatus.demandForecast]);
```

## How to Verify It's Using the RL Model

1. **Check the reason text:**
   - 🤖 = RL Model prediction
   - 📊 = Fallback calculation

2. **Check browser console (F12):**
   - See network request to `http://localhost:5000/api/recommend`
   - Response shows `"source": "rl_model"` or `"source": "fallback"`

3. **Check API terminal:**
   - See incoming requests logged
   - See RL model prediction output

## Architecture Now:

```
User → React Dashboard (http://localhost:3000)
          ↓ fetch()
       Flask API (http://localhost:5000)
          ↓ torch.load()
       PyTorch RL Model (ppo_store_agents_gym.pt)
          ↓ model.predict()
       Order Quantity Decision
          ↓ JSON response
       Dashboard displays 🤖 RL recommendation!
```

## Notes

- If API server is down, dashboard automatically falls back to client-side calculations
- Recommendation reason will show 🤖 (RL) or 📊 (fallback) to indicate source
- All existing dashboard features still work (what-if, scenarios, etc.)
