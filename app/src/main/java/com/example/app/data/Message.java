package com.example.app.data;

import jakarta.persistence.*;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "messages")
public class Message {

    @Id
    @GeneratedValue
    private Long id;

    @Column(nullable = false)
    private String content;

    @Enumerated(EnumType.STRING)
    private MessageType type; // TEXT, FILE

    @ManyToOne
    @JoinColumn(name = "sender_id", nullable = false)
    private User sender;

    // wiadomość prywatna
    @ManyToOne
    @JoinColumn(name = "receiver_id")
    private User receiver;

    // chat projektowy
    @ManyToOne
    @JoinColumn(name = "project_id")
    private Project project;

    @OneToOne
    private FileEntity file;

    private LocalDateTime createdAt;
}