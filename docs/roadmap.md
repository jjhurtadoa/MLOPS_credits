## 🚀 Mejoras Futuras

### Corto plazo

- **Variables de entorno por ambiente**:  Configuración separada para dev/qa/prod con archivos `.env` específicos para cada entorno (actualmente solo existe ambiente local)
- **Sistema de alertas**: Implementar Alertmanager con Prometheus para notificaciones automáticas (latencia alta, errores, API caída)
- **Pruebas de stress**:  Validar capacidad de carga con herramientas como Locust o k6 (usuarios concurrentes, throughput, SLAs)

### Mediano plazo

- **Monitoreo de drift**: Integrar Evidently AI o similar para detectar data drift y model drift en producción
- **Deploy en servidor**: Desplegar en VPS (DigitalOcean, Linode) o plataforma cloud (Render, Railway, AWS ECS) con CI/CD automático
- **A/B Testing**: Implementar framework para comparar versiones de modelos en producción (traffic splitting, métricas comparativas)

### Largo plazo

- **Orquestación con Airflow/Prefect**: Automatizar pipelines de reentrenamiento periódico basado en métricas o calendario
- **Auto-scaling**:  Configurar escalado automático basado en métricas de carga (Kubernetes, Docker Swarm, AWS ECS)
- **Feature store**: Centralizar features para reutilización entre múltiples modelos (Feast, cache con Redis)
- **DVC para datasets grandes**: Versionado de datos con DVC cuando el volumen de datos escale (actualmente innecesario con Boston Housing)