package com.example.app.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.Set;

@Data
public class ProjectDto {
    @JsonProperty("project_id")
    private Long projectId;

    private String name;

    private Set<UserDto> users;
}
