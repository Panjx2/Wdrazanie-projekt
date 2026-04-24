package com.example.app.dto;

import com.example.app.data.MessageType;

import java.time.LocalDateTime;

public record MessageDTO(Long id,
                         Long senderId,
                         String senderUsername,
                         Long receiverId,
                         Long projectId,
                         String content,
                         MessageType type,
                         Long fileId,
                         String fileName,
                         LocalDateTime createdAt) {
}
