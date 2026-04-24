package com.example.app.service;


import com.example.app.data.*;
import com.example.app.dto.MessageDTO;
import com.example.app.repository.FileEntityRepository;
import com.example.app.repository.MessageRepository;
import com.example.app.repository.ProjectRepository;
import com.example.app.repository.UserRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;

@Service
public class MessageService {

    private final MessageRepository messageRepository;
    private final UserRepository userRepository;
    private final ProjectRepository projectRepository;
    private final FileEntityRepository fileRepository;

    public MessageService(MessageRepository messageRepository,
                          UserRepository userRepository,
                          ProjectRepository projectRepository,
                          FileEntityRepository fileRepository) {
        this.messageRepository = messageRepository;
        this.userRepository = userRepository;
        this.projectRepository = projectRepository;
        this.fileRepository = fileRepository;
    }


    public MessageDTO savePrivateMessage(MessageDTO dto) {

        User sender = userRepository.findById(dto.senderId()).orElseThrow();
        User receiver = userRepository.findById(dto.receiverId()).orElseThrow();

        Message message = new Message();
        message.setSender(sender);
        message.setReceiver(receiver);
        message.setContent(dto.content());
        message.setType(dto.type());
        message.setCreatedAt(LocalDateTime.now());

        if (dto.type() == MessageType.FILE && dto.fileId() != null) {
            FileEntity file = fileRepository.findById(dto.fileId()).orElseThrow();
            message.setFile(file);
        }

        Message saved = messageRepository.save(message);

        return toDto(saved);
    }

    public MessageDTO saveProjectMessage(MessageDTO dto) {

        User sender = userRepository.findById(dto.senderId()).orElseThrow();
        Project project = projectRepository.findById(dto.projectId()).orElseThrow();

        Message message = new Message();
        message.setSender(sender);
        message.setProject(project);
        message.setContent(dto.content());
        message.setType(dto.type());
        message.setCreatedAt(LocalDateTime.now());

        Message saved = messageRepository.save(message);

        return toDto(saved);
    }


    public static MessageDTO toDto(Message m) {
        return new MessageDTO(
                m.getId(),
                m.getSender().getId(),
                m.getSender().getUsername(),
                m.getReceiver() != null ? m.getReceiver().getId(): null,
                m.getProject() != null ? m.getProject().getId() : null,
                m.getContent(),
                m.getType(),
                m.getFile() != null ? m.getFile().getId() : null,
                m.getFile() != null ? m.getFile().getName() : null,
                m.getCreatedAt()
        );
    }
}