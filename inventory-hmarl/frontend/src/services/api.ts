/**
 * API Service for RL Model Integration
 * 
 * This service handles communication between the React frontend
 * and the Flask backend serving the trained RL model.
 */

const API_BASE_URL = 'http://localhost:5000';

export interface RecommendationResponse {
    source: 'rl_model' | 'fallback';
    orderQuantity: number;
    reason: string;
    expectedServiceLevel: number;
    stockoutRisk: 'LOW' | 'MEDIUM' | 'HIGH';
    estimatedCost: number;
}

export interface InventoryStatusResponse {
    inventory: number;
    reorderPoint: number;
    daysOfSupply: number;
    demandForecast: number;
    safetyStock: number;
    inTransit: number;
}

/**
 * Get inventory recommendation from RL model
 */
export async function getRecommendation(
    store: string,
    sku: string,
    scenario: string
): Promise<RecommendationResponse> {
    const url = `${API_BASE_URL}/api/recommend?store=${store}&sku=${sku}&scenario=${scenario}`;

    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`API request failed: ${response.statusText}`);
    }

    return await response.json();
}

/**
 * Get current inventory status
 */
export async function getInventoryStatus(
    store: string,
    sku: string
): Promise<InventoryStatusResponse> {
    const url = `${API_BASE_URL}/api/inventory/status?store=${store}&sku=${sku}`;

    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`API request failed: ${response.statusText}`);
    }

    return await response.json();
}

/**
 * Check if API server is healthy
 */
export async function checkHealth(): Promise<boolean> {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        return response.ok;
    } catch {
        return false;
    }
}
