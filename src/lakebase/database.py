import uuid
import logging
from datetime import datetime
from databricks.sdk import WorkspaceClient
import os
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
import string
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

class LakebaseDatabase:
    def __init__(self, host: str):
        self.connection_pool = None
        self.host = host
        self.client_id = os.getenv("DATABRICKS_CLIENT_ID")
        self.client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")
        self.w = WorkspaceClient(host=self.host, client_id=self.client_id, client_secret=self.client_secret)

    def get_connection_pool(self):
        return self.connection_pool
    
    def initialize_connection(self, user: str | None = None, instance_name: str | None = None):
        """Initialize database connection using Databricks credentials"""
        instance = self.w.database.get_database_instance(name=instance_name)
        try:
            # Database connection setup
            try:
                cred = self.w.database.generate_database_credential(request_id=str(uuid.uuid4()), instance_names=[instance_name])
                host = instance.read_write_dns
                port = 5432
                database = "databricks_postgres"
                password = cred.token
                conn_string = f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode=require"

                # Create custom connection class with access to workspace client
                CustomConnection = self._create_connection_factory(instance_name)

                self.connection_pool = ConnectionPool(
                    conninfo=f"dbname={database} user={user} host={host} sslmode=require",
                    connection_class=CustomConnection,
                    min_size=1,
                    max_size=10,
                    open=True
                )
                # Create table if it doesn't exist
                # with self.connection_pool.connect() as conn:
                #     # TODO: Need to provide create and insert permissions to the SP? 
                #     checkpointer = PostgresSaver(conn)
                #     # NOTE: you need to call .setup() the first time you're using your checkpointer
                #     checkpointer.setup()
                #     logger.info("PostgresSaver setup completed successfully.")
                return conn_string
            except Exception as e:
                logger.error(f"Error connecting to Postgres or setting up PostgresSaver: {str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"Error saving conversation to database: {str(e)}", exc_info=True)
        

    def get_conversation_history(self, session_id=None, limit=100):
        """Retrieve conversation history from database"""
        return None  # Placeholder for actual implementation
    
    def _create_connection_factory(self, instance_name: str):
        """Create a connection factory that captures the workspace client and instance name"""
        workspace_client = self.w
        
        class CustomConnection(psycopg.Connection):
            @classmethod
            def connect(cls, conninfo='', **kwargs):
                # Generate fresh credentials for this connection
                cred = workspace_client.database.generate_database_credential(
                    request_id=str(uuid.uuid4()), 
                    instance_names=[instance_name]
                )
                kwargs['password'] = cred.token
                
                # Call the superclass's connect method with updated kwargs
                return super().connect(conninfo, **kwargs)
        
        return CustomConnection
