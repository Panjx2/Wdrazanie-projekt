package com.example.app.controller;

import com.example.app.dto.MessageDTO;
import com.example.app.service.MessageService;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Controller;

@Controller
public class ChatController {

    private final SimpMessagingTemplate messagingTemplate;
    private final MessageService messageService;

    public ChatController(SimpMessagingTemplate messagingTemplate,
                          MessageService messageService) {
        this.messagingTemplate = messagingTemplate;
        this.messageService = messageService;
    }


    @MessageMapping("/chat.private")
    public void sendPrivate(MessageDTO message) {

        MessageDTO saved = messageService.savePrivateMessage(message);

        messagingTemplate.convertAndSendToUser(
                saved.receiverId().toString(),
                "/queue/messages",
                saved
        );
    }


    @MessageMapping("/chat.project")
    public void sendToProject(MessageDTO message) {

        MessageDTO saved = messageService.saveProjectMessage(message);

        messagingTemplate.convertAndSend(
                "/topic/project/" + saved.projectId(),
                saved
        );
    }
}