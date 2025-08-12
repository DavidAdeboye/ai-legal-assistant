async def process_document_background(document_id: str, file_path: str, file_extension: str):
    """Background task to process document"""
    try:
        # Extract text based on file type
        if file_extension == 'pdf':
            chunks = await extract_text_from_pdf(file_path)
        elif file_extension == 'docx':
            chunks = await extract_text_from_docx(file_path)
        else:
            raise Exception(f"Unsupported file type: {file_extension}")
        # Store chunks with embeddings
        await store_document_chunks(document_id, chunks)
        logger.info(f"Successfully processed document {document_id} with {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        conn = await get_db_connection()
        if conn:
            try:
                await conn.execute(
                    '''UPDATE documents SET status = 'failed' WHERE id = $1''',
                    document_id
                )
            finally:
                await release_db_connection(conn)
        else:
            if document_id in memory_storage["documents"]:
                memory_storage["documents"][document_id]["status"] = "failed"
    finally:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
