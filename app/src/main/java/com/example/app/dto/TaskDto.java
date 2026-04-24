package com.example.app.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.example.app.data.TaskStatus;
import lombok.Data;

@Data
public class TaskDto {
    @JsonProperty("task_id")
    private Long taskId;

    private String name;
    private String priority;
    private TaskStatus status;
    private UserDto assignedUser;
    private ProjectDto project;
}
