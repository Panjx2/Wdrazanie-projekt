package com.project.model;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class Task {
    @JsonProperty("task_id")
    @JsonAlias("taskId")
    private Long taskId;

    private String name;
    private String priority;
    private TaskStatus status;
    private User assignedUser;
    private Project project;

    @JsonIgnore
    private Long projectId;

    @JsonIgnore
    private Long assignedUserId;
}
