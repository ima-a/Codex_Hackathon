# HMARL Supply Chain Dashboard

A modern, responsive dashboard for visualizing Hierarchical Multi-Agent Reinforcement Learning (HMARL) results in retail supply chain optimization.

## 🎯 Overview

This dashboard visualizes the performance comparison between baseline (rule-based) policies and PPO-trained agents across a multi-echelon supply chain system.

**Key Results**:
- ✅ **Service Level**: 91.2% → 97.2% (+6.6%)
- ✅ **Stockouts**: 450 → 150 units (-66.7%)
- ✅ **Holding Costs**: $12K → $9.5K (-20.8%)
- ✅ **Avg Reward**: 0 → 300

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ and npm

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The application will open at `http://localhost:3000`

### Build for Production

```bash
npm run build
npm run preview
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Navigation.tsx          # Top navigation bar
│   │   ├── KPICard.tsx             # Metric comparison card
│   │   └── ComparisonChart.tsx     # Bar/line charts
│   ├── pages/
│   │   ├── Landing.tsx             # Overview & architecture
│   │   ├── Dashboard.tsx           # Main results (KPIs + charts)
│   │   ├── AgentBehavior.tsx       # Agent explainability
│   │   └── Reconciliation.tsx      # Metrics comparison
│   ├── App.tsx                     # Main app with routing
│   └── main.tsx                    # Entry point
├── package.json
├── vite.config.ts
└── tailwind.config.js
```

## 📊 Pages

### 1. Landing Page (`/`)
- Hero section with project overview
- Feature cards explaining the system
- Architecture diagram
- Call-to-action

### 2. Dashboard (`/dashboard`) ⭐ MAIN PAGE
- **KPI Cards**: 4 key metrics with baseline vs PPO comparison
  - Service Level
  - Stockouts
  - Holding Costs
  - Avg Reward
- **Charts**: Visual comparisons
  - Service Level (Bar chart)
  - Stockouts (Bar chart)
  - Holding Costs (Bar chart)
  - Training Progression (Line chart)
- **Key Insights**: Summary of improvements

### 3. Agent Behavior (`/agents`)
- Tabbed interface for 3 agent types
- Observations & Actions explained
- Learning method details
- Performance metrics

### 4. Reconciliation (`/reconciliation`)
- Plain English explanation
- Sample reconciliation table
- Deviation reason codes
- Business value explanation

## 🎨 Technology Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Fast build tool
- **Tailwind CSS** - Styling
- **Recharts** - Charts & graphs
- **React Router** - Navigation

## 📈 Mock Data

The dashboard currently uses mock data embedded in the components. To integrate with real backend data:

1. Update `Dashboard.tsx` to fetch from API:
```typescript
useEffect(() => {
  fetch('/api/baseline_metrics')
    .then(res => res.json())
    .then(data => setBaselineMetrics(data));
}, []);
```

2. Or load static JSON files from `/public/data/`:
```typescript
fetch('/data/baseline_metrics.json')
  .then(res => res.json())
  .then(data => setBaselineMetrics(data));
```

## 🎯 Design Goals

✅ **Non-Technical Friendly**: No ML jargon on main screens  
✅ **< 60 Second Understanding**: Judge can grasp improvements quickly  
✅ **Professional**: Clean, modern design  
✅ **Responsive**: Works on laptops (demo-ready)  
✅ **Explainable**: Clear connection between AI decisions and business outcomes  

## 🛠️ Customization

### Colors

Edit `tailwind.config.js`:
```javascript
colors: {
  primary: '#3B82F6',    // Blue
  success: '#10B981',    // Green
  warning: '#F59E0B',    // Yellow
  danger: '#EF4444',     // Red
}
```

### Metrics

Update values in `Dashboard.tsx`:
```typescript
setBaselineMetrics({
  service_level: 0.912,
  stockouts: 450,
  holding_cost: 12000,
  avg_reward: 0,
});
```

## 📝 Scripts

- `npm run dev` - Start development server (port 3000)
- `npm run build` - Build for production
- `npm run preview` - Preview production build

## 🎬 Demo Tips

For hackathon judges:

1. Start at Landing page - quick 30-second overview
2. Jump to Dashboard - show the core improvements
3. Optional: Agent Behavior for technical judges
4. End with Reconciliation - explain business value

**Target Demo Time**: 3-5 minutes

## 📦 Deployment

### Static Deployment (Netlify, Vercel)

```bash
npm run build
# Upload 'dist' folder
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
CMD ["npm", "run", "preview"]
```

## 🐛 Troubleshooting

**Charts not rendering?**
- Ensure Recharts is installed: `npm install recharts`

**Routing not working after build?**
- Configure your server to redirect all routes to `index.html`

**Styling issues?**
- Run `npm install -D tailwindcss postcss autoprefixer`
- Ensure `index.css` imports are correct

## 📄 License

MIT

## 👥 Contributors

Built for Codex Hackathon 2026

---

**Built with ❤️ for intelligent supply chain management**
