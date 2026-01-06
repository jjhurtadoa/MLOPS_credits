# Grafana — Quick Guide

> Visualiza métricas de Prometheus y crea dashboards para el servicio y el modelo.

## UI rápida
- URL: http://localhost:3000
- Usuario/Pass por defecto: `admin` / `admin`

---

## Objetivo
Crear paneles que muestren salud del servicio (latencia, errores, throughput) y calidad del modelo (error, drift, distribución de predicciones).

---

## Panels recomendados
- Latencia por endpoint (p50/p95/p99). Ejemplo PromQL:
  `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint))`
- Throughput: peticiones/sec por endpoint. Ejemplo:
  `rate(http_requests_total[5m])`
- Tasa de errores por endpoint (`rate(http_requests_total{status=~"5.."}[5m])`)
- Histogramas por feature o por latencia para detectar drift/regresiones
- Anotar releases/model-version y crear alertas sobre p95 latencia o subida de errores

---

## Ejemplo rápido: crear panel de throughput
1. Add → Dashboard → Add new panel
2. Datasource: Prometheus
3. Query: `rate(http_requests_total[5m])`
4. Type: Time series, Unit: ops/s

---

## Panel de latencia (p95)
Query ejemplo:
```promql
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint))
```

## Ver ejemplos (imágenes)

![Dashboard general](../../images/grafana-01-dashboard.png)

![Panel detalle](../../images/grafana-02-detail-status-code.png)

---

## Buenas prácticas
- Anotar releases/model-version en Grafana (tags/time range) para correlacionar cambios.
- Usar variables (dashboard variables) para seleccionar `model` o `version`.
- Crear alertas basadas en métricas clave (p95 latency, error increase, drift threshold).
- Guardar dashboards como JSON y versionarlos en el repo.

---

## Enlaces útiles
- Grafana docs: https://grafana.com/docs/
- Dashboards export/import: https://grafana.com/docs/grafana/latest/dashboards/

---

[← Volver a Prometheus](prometheus.md) | [Volver al índice](../index.md)