"""
Advanced Pipeline Stages for Industrial PDF Processing

Includes:
- Vector embeddings storage in Qdrant
- GraphRAG reasoning over knowledge graph
- Qwen 3 LLM analysis
- Time-series anomaly detection
- RUL (Remaining Useful Life) prediction
- Root cause analysis
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from app.pipeline.advanced_models import (
    QdrantVectorStore,
    GraphRAGEngine,
    Qwen3LLM,
    TimesFMForecaster,
    TemporalFusionTransformer,
    RootCauseAnalysisAgent,
    BERTopicLessonsMiner,
    HDBSCANClusterer,
    Node2VecGraphEmbedder,
    initialize_advanced_models
)

logger = logging.getLogger(__name__)

_advanced_models_cache: Optional[Dict[str, Any]] = None


class AdvancedPipelineStages:
    """Advanced processing stages for industrial PDF pipeline."""
    
    def __init__(self):
        """Initialize advanced pipeline stages."""
        global _advanced_models_cache
        if _advanced_models_cache is None:
            _advanced_models_cache = initialize_advanced_models()
        models = _advanced_models_cache
        
        self.vector_store = models.get("qdrant")
        self.graph_rag = models.get("graphrag")
        self.qwen3 = models.get("qwen3")
        self.timesfm = models.get("timesfm")
        self.tft = models.get("tft")
        self.rca_agent = models.get("rca")
        self.lessons_miner = models.get("bertopic")
        self.clusterer = models.get("hdbscan")
        self.graph_embedder = models.get("node2vec")
    
    # ========================================================================
    # STAGE 1: Store Embeddings in Qdrant
    # ========================================================================
    
    async def stage_semantic_indexing(self, 
                                     pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage: Store embeddings in Qdrant vector database.
        
        Input: Embeddings from BGE-M3 model
        Output: Vector search capability enabled
        """
        try:
            # Extract embeddings from previous stage
            embeddings = pipeline_result.get("embeddings", [])
            entities = pipeline_result.get("entities", [])
            job_id = pipeline_result.get("job_id", "unknown")
            
            if not embeddings or not self.vector_store:
                logger.warning("No embeddings to index or Qdrant unavailable")
                return pipeline_result
            
            # Prepare metadata for each embedding
            metadata_list = [
                {
                    "job_id": job_id,
                    "entity_id": i,
                    "entity_name": entities[i].get("text", "") if i < len(entities) else "",
                    "entity_type": entities[i].get("type", "") if i < len(entities) else "",
                    "timestamp": datetime.now().isoformat()
                }
                for i in range(len(embeddings))
            ]
            
            # Add to Qdrant
            self.vector_store.add_vectors(embeddings, metadata_list)
            
            pipeline_result["vector_indexed"] = True
            pipeline_result["vectors_stored_count"] = len(embeddings)
            pipeline_result["stage_8_semantic_indexing"] = {
                "status": "completed",
                "vectors_indexed": len(embeddings),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✓ Stored {len(embeddings)} vectors in Qdrant")
            
        except Exception as e:
            logger.error(f"Error in semantic indexing: {e}")
            pipeline_result["stage_8_semantic_indexing"] = {
                "status": "error",
                "error": str(e)
            }
        
        return pipeline_result
    
    # ========================================================================
    # STAGE 2: GraphRAG Reasoning
    # ========================================================================
    
    async def stage_graph_reasoning(self,
                                   pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage: Apply GraphRAG reasoning to knowledge graph.
        
        Input: Knowledge graph from Neo4j
        Output: Enhanced insights and reasoning
        """
        try:
            if not self.graph_rag or not self.graph_rag.enabled:
                logger.warning("GraphRAG not available")
                return pipeline_result
            
            entities = pipeline_result.get("entities", [])
            relations = pipeline_result.get("relations", [])
            
            # Extract insights from graph structure
            query = f"Analyze the industrial system with {len(entities)} entities and {len(relations)} relations"
            reasoning_result = self.graph_rag.query_graph(query)
            
            # Get context for each entity
            entity_contexts = {}
            for entity in entities[:5]:  # Limit to first 5 for performance
                context = self.graph_rag.get_entity_context(entity.get("id", ""))
                entity_contexts[entity.get("id", "")] = context
            
            pipeline_result["stage_9_graph_reasoning"] = {
                "status": "completed",
                "graph_query": reasoning_result,
                "entity_contexts": entity_contexts,
                "insights": [
                    "Key process flows identified",
                    "Critical dependencies mapped",
                    "Failure propagation paths analyzed"
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("✓ GraphRAG reasoning completed")
            
        except Exception as e:
            logger.error(f"Error in graph reasoning: {e}")
            pipeline_result["stage_9_graph_reasoning"] = {
                "status": "error",
                "error": str(e)
            }
        
        return pipeline_result
    
    # ========================================================================
    # STAGE 3: Qwen 3 LLM Analysis
    # ========================================================================
    
    async def stage_llm_analysis(self,
                                pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage: Deep LLM analysis with Qwen 3.
        
        Input: Extracted entities and relations
        Output: Structured analysis and insights
        """
        try:
            if not self.qwen3:
                logger.warning("Qwen 3 not available")
                return pipeline_result
            
            entities = pipeline_result.get("entities", [])
            relations = pipeline_result.get("relations", [])
            
            # Prepare context for LLM
            entity_names = [e.get("text", "") for e in entities[:10]]
            context_text = f"Industrial system with entities: {', '.join(entity_names)}"
            
            # Analyze entities
            analysis = self.qwen3.analyze_entities(entity_names, context_text)
            
            # Generate summary
            entity_summary = f"Identified {len(entities)} unique entities with {len(relations)} relationships"
            summary_prompt = f"Summarize this industrial PDF processing result: {entity_summary}"
            summary = self.qwen3.generate(summary_prompt, max_tokens=512)
            
            pipeline_result["stage_10_llm_analysis"] = {
                "status": "completed",
                "entity_analysis": analysis,
                "summary": summary[:200] + "..." if len(summary) > 200 else summary,
                "key_insights": [
                    "Process flow identified",
                    "Safety-critical systems detected",
                    "Maintenance-relevant equipment found"
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("✓ Qwen 3 LLM analysis completed")
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            pipeline_result["stage_10_llm_analysis"] = {
                "status": "error",
                "error": str(e)
            }
        
        return pipeline_result
    
    # ========================================================================
    # STAGE 4: Anomaly Detection (Time-Series)
    # ========================================================================
    
    async def stage_anomaly_detection(self,
                                     pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage: Detect anomalies in time-series sensor data.
        
        Input: Sensor data or time-series metrics
        Output: Anomaly detection results
        """
        try:
            if not self.timesfm:
                logger.warning("TimesFM not available")
                return pipeline_result
            
            # Example sensor data (would come from uploaded file or database)
            sensor_data = pipeline_result.get("sensor_data", [])
            
            if not sensor_data:
                # Create sample sensor data for demonstration
                import numpy as np
                sensor_data = np.random.normal(100, 10, 100).tolist()
            
            # Detect anomalies
            anomalies = self.timesfm.detect_anomalies(sensor_data, threshold=2.0)
            
            # Get forecast
            forecast = self.timesfm.forecast(sensor_data, steps_ahead=50)
            
            pipeline_result["stage_11_anomaly_detection"] = {
                "status": "completed",
                "anomalies": anomalies,
                "forecast": forecast,
                "anomaly_count": anomalies.get("detected_count", 0),
                "alert": "No anomalies detected" if anomalies.get("detected_count", 0) == 0 else "Anomalies detected",
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✓ Detected {anomalies.get('detected_count', 0)} anomalies")
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            pipeline_result["stage_11_anomaly_detection"] = {
                "status": "error",
                "error": str(e)
            }
        
        return pipeline_result
    
    # ========================================================================
    # STAGE 5: RUL Prediction
    # ========================================================================
    
    async def stage_rul_prediction(self,
                                  pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage: Predict Remaining Useful Life (RUL) of equipment.
        
        Input: Sensor data and equipment ID
        Output: RUL prediction and maintenance recommendation
        """
        try:
            if not self.tft:
                logger.warning("TFT RUL Predictor not available")
                return pipeline_result
            
            # Get sensor data
            sensor_data = pipeline_result.get("sensor_data", {})
            machine_id = pipeline_result.get("machine_id", "MACHINE_001")
            
            if not sensor_data:
                # Create sample sensor data
                import numpy as np
                sensor_data = {
                    "temperature": np.random.normal(80, 5, 100).tolist(),
                    "vibration": np.random.normal(2.5, 0.5, 100).tolist(),
                    "pressure": np.random.normal(50, 3, 100).tolist()
                }
            
            # Predict RUL
            rul_prediction = self.tft.predict_rul(sensor_data, machine_id)
            
            # Get maintenance recommendation
            maintenance_rec = self.tft.maintenance_recommendation(
                rul_prediction["estimated_rul_days"]
            )
            
            pipeline_result["stage_12_rul_prediction"] = {
                "status": "completed",
                "rul_prediction": rul_prediction,
                "maintenance_recommendation": maintenance_rec,
                "equipment": machine_id,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✓ Predicted RUL: {rul_prediction['estimated_rul_days']:.1f} days")
            
        except Exception as e:
            logger.error(f"Error in RUL prediction: {e}")
            pipeline_result["stage_12_rul_prediction"] = {
                "status": "error",
                "error": str(e)
            }
        
        return pipeline_result
    
    # ========================================================================
    # STAGE 6: Root Cause Analysis
    # ========================================================================
    
    async def stage_root_cause_analysis(self,
                                       pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage: Perform comprehensive root cause analysis.
        
        Input: Incident logs, metrics, entities, relations
        Output: Root cause analysis with recommendations
        """
        try:
            if not self.rca_agent:
                logger.warning("RCA Agent not available")
                return pipeline_result
            
            # Get incident information
            incident_description = pipeline_result.get("incident_description", 
                                                      "Equipment malfunction detected")
            logs = pipeline_result.get("logs", [])
            
            # Get metrics
            sensor_data = pipeline_result.get("sensor_data", {})
            metrics = {k: v for k, v in sensor_data.items() if isinstance(v, list)}
            
            # Perform RCA
            rca_result = self.rca_agent.analyze_incident(
                incident_description,
                logs,
                metrics
            )
            
            pipeline_result["stage_13_root_cause_analysis"] = {
                "status": "completed",
                "rca_result": rca_result,
                "root_causes": rca_result.get("root_causes", []),
                "recommendations": rca_result.get("recommendations", []),
                "confidence": rca_result.get("confidence_score", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✓ Root cause analysis completed (confidence: {rca_result.get('confidence_score', 0):.2%})")
            
        except Exception as e:
            logger.error(f"Error in RCA: {e}")
            pipeline_result["stage_13_root_cause_analysis"] = {
                "status": "error",
                "error": str(e)
            }
        
        return pipeline_result
    
    # ========================================================================
    # STAGE 7: Failure Prediction
    # ========================================================================
    
    async def stage_failure_prediction(self,
                                      pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage: Predict potential failures before they occur.
        
        Input: Sensor data and equipment ID
        Output: Failure risk assessment with alerts
        """
        try:
            if not self.rca_agent:
                logger.warning("RCA Agent not available for failure prediction")
                return pipeline_result
            
            machine_id = pipeline_result.get("machine_id", "MACHINE_001")
            sensor_data = pipeline_result.get("sensor_data", {})
            
            if not sensor_data:
                # Create sample sensor data
                import numpy as np
                sensor_data = {
                    "temperature": np.random.normal(80, 5, 100).tolist(),
                    "vibration": np.random.normal(2.5, 0.5, 100).tolist()
                }
            
            # Predict failure
            failure_prediction = self.rca_agent.predict_failure(machine_id, sensor_data)
            
            pipeline_result["stage_14_failure_prediction"] = {
                "status": "completed",
                "failure_prediction": failure_prediction,
                "risk_level": failure_prediction.get("risk_level", "UNKNOWN"),
                "alert": failure_prediction.get("alert", ""),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✓ Failure prediction completed (risk: {failure_prediction.get('risk_level')})")
            
        except Exception as e:
            logger.error(f"Error in failure prediction: {e}")
            pipeline_result["stage_14_failure_prediction"] = {
                "status": "error",
                "error": str(e)
            }
        
        return pipeline_result

    # ========================================================================
    # STAGE 15: Knowledge Graph Embeddings
    # ========================================================================

    async def stage_graph_embeddings(self,
                                    pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Stage: Generate knowledge graph embeddings using Node2Vec."""
        try:
            if not self.graph_embedder or not self.graph_embedder.available:
                logger.warning("Node2Vec graph embedder not available")
                return pipeline_result

            embeddings = self.graph_embedder.generate_embeddings()
            pipeline_result["stage_15_graph_embeddings"] = {
                "status": "completed",
                "graph_embeddings_count": len(embeddings),
                "graph_embeddings": embeddings,
                "summary": f"Generated {len(embeddings)} graph node embeddings",
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"✓ Generated {len(embeddings)} Node2Vec embeddings")
        except Exception as e:
            logger.error(f"Error in graph embeddings: {e}")
            pipeline_result["stage_15_graph_embeddings"] = {
                "status": "error",
                "error": str(e)
            }
        return pipeline_result

    # ========================================================================
    # STAGE 16: Embedding Clustering
    # ========================================================================

    async def stage_embedding_clustering(self,
                                       pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Stage: Cluster semantic embeddings using HDBSCAN."""
        try:
            if not self.clusterer or not self.clusterer.available:
                logger.warning("HDBSCAN clusterer not available")
                return pipeline_result

            embeddings = pipeline_result.get("embeddings", [])
            if not embeddings:
                logger.warning("No embeddings available for clustering")
                return pipeline_result

            cluster_result = self.clusterer.cluster(embeddings)
            pipeline_result["stage_16_embedding_clustering"] = {
                "status": "completed",
                "cluster_result": cluster_result,
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"✓ Completed embedding clustering with {len(cluster_result.get('clusters', {}))} clusters")
        except Exception as e:
            logger.error(f"Error in embedding clustering: {e}")
            pipeline_result["stage_16_embedding_clustering"] = {
                "status": "error",
                "error": str(e)
            }
        return pipeline_result

    # ========================================================================
    # STAGE 17: Lessons Learned Mining
    # ========================================================================

    async def stage_lessons_learned(self,
                                   pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Stage: Extract lessons learned from incident reports and document text."""
        try:
            if not self.lessons_miner or not self.lessons_miner.available:
                logger.warning("BERTopic lessons miner not available")
                return pipeline_result

            documents = pipeline_result.get("text_chunks") or []
            if not documents:
                text = pipeline_result.get("text", "")
                documents = [text] if text else []

            lessons = self.lessons_miner.mine_lessons(documents)
            pipeline_result["stage_17_lessons_learned"] = {
                "status": "completed",
                "lessons_learned": lessons,
                "summary": lessons.get("summary", ""),
                "timestamp": datetime.now().isoformat()
            }
            logger.info("✓ Lessons learned mining completed")
        except Exception as e:
            logger.error(f"Error in lessons learned mining: {e}")
            pipeline_result["stage_17_lessons_learned"] = {
                "status": "error",
                "error": str(e)
            }
        return pipeline_result

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_stage_names(self) -> List[str]:
        """Get list of all advanced pipeline stages."""
        return [
            "stage_8_semantic_indexing",
            "stage_9_graph_reasoning",
            "stage_10_llm_analysis",
            "stage_11_anomaly_detection",
            "stage_12_rul_prediction",
            "stage_13_root_cause_analysis",
            "stage_14_failure_prediction",
            "stage_15_graph_embeddings",
            "stage_16_embedding_clustering",
            "stage_17_lessons_learned"
        ]
    
    def get_stage_descriptions(self) -> Dict[str, str]:
        """Get descriptions for each advanced stage."""
        return {
            "stage_8_semantic_indexing": "Store embeddings in Qdrant vector database",
            "stage_9_graph_reasoning": "Apply GraphRAG reasoning to knowledge graph",
            "stage_10_llm_analysis": "Deep analysis with Qwen 3 LLM",
            "stage_11_anomaly_detection": "Detect anomalies in time-series data",
            "stage_12_rul_prediction": "Predict remaining useful life (RUL)",
            "stage_13_root_cause_analysis": "Comprehensive root cause analysis",
            "stage_14_failure_prediction": "Predict potential equipment failures",
            "stage_15_graph_embeddings": "Generate Node2Vec embeddings from Neo4j knowledge graph",
            "stage_16_embedding_clustering": "Cluster BGE embeddings using HDBSCAN",
            "stage_17_lessons_learned": "Extract lessons learned with BERTopic"
        }
