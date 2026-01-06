# Prometheus — Monitorización de métricas

Prometheus es la herramienta para *revisar* y consultar métricas de tiempo (time-series) expuestas por la API y los procesos de inferencia. Se usa principalmente para inspección, visualización en Grafana y para disparar alertas cuando las métricas cruzan umbrales.

---

## Acceso rápido
- UI: http://localhost:9090
- Verifica targets en: Status → Targets (debe aparecer tu servicio `/metrics`).

---

## ¿Para qué lo usamos aquí?
- Ver latencia y disponibilidad de los endpoints (p50/p95/p99).
- Monitorizar tiempos de inferencia y throughput del modelo.
- Detectar errores de predicción y anomalías en distribuciones (drift).
- Servir métricas que Grafana consume para dashboards y alertas.

---

## Archivos relevantes
- La configuración de scrape está en `monitoring/prometheus.yml`.

---

## Imágenes de referencia
Targets (verificación):

![Prometheus Targets](../../images/prometheus-01-target.png)

Ejemplo de query/visualización:

![Prometheus Graph](../../images/prometheus-02-graph.png)

---

> Nota: Levantamos Prometheus usando la imagen oficial de Docker.
>
>
> Prometheus recoge las métricas; no instrumenta el servicio. Para exportar métricas desde FastAPI o procesos de inferencia usa un cliente Prometheus (librería `prometheus_client`) y expón `/metrics`.

[← Volver](../index.md) | [Grafana →](grafana.md)