import uuid
import logging
from datetime import datetime
from urllib.parse import quote_plus
from databricks.sdk import WorkspaceClient
import os
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from psycopg_pool import ConnectionPool
from databricks.sdk.service.postgres import EndpointType

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

    def initialize_connection(self, user: str | None = None, project_name: str | None = None):
        """Initialize database connection using Databricks Lakebase Autoscaling credentials"""
        # Get the project's production branch primary endpoint
        endpoints = list(self.w.postgres.list_endpoints(
            parent=f"projects/{project_name}/branches/production"
        ))
        primary_endpoint = next(
            ep for ep in endpoints
            if ep.status.endpoint_type == EndpointType.ENDPOINT_TYPE_READ_WRITE
        )

        endpoint_name = primary_endpoint.name
        host = primary_endpoint.status.hosts.host

        try:
            try:
                # Generate ephemeral credentials via Postgres API
                cred = self.w.postgres.generate_database_credential(endpoint=endpoint_name)
                port = 5432
                database = "databricks_postgres"
                password = cred.token
                # URL-encode user and password to handle special characters (e.g. @ in email)
                conn_string = f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{database}?sslmode=require"

                # Create custom connection class with token refresh
                CustomConnection = self._create_connection_factory(endpoint_name)

                self.connection_pool = ConnectionPool(
                    conninfo=f"dbname={database} user={user} host={host} sslmode=require",
                    connection_class=CustomConnection,
                    min_size=1,
                    max_size=10,
                    open=True
                )
                return conn_string
            except Exception as e:
                logger.error(f"Error connecting to Postgres or setting up PostgresSaver: {str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"Error saving conversation to database: {str(e)}", exc_info=True)


    def get_conversation_history(self, session_id=None, limit=100):
        """Retrieve conversation history from database"""
        return None  # Placeholder for actual implementation

    def _create_connection_factory(self, endpoint_name: str):
        """Create a connection factory that captures the workspace client and endpoint name"""
        workspace_client = self.w

        class CustomConnection(psycopg.Connection):
            @classmethod
            def connect(cls, conninfo='', **kwargs):
                # Generate fresh credentials for this connection
                cred = workspace_client.postgres.generate_database_credential(
                    endpoint=endpoint_name
                )
                kwargs['password'] = cred.token

                # Call the superclass's connect method with updated kwargs
                return super().connect(conninfo, **kwargs)

        return CustomConnection
