package com.example.app.mapper;

import com.example.app.data.Project;
import com.example.app.data.Task;
import com.example.app.data.User;
import com.example.app.dto.ProjectDto;
import com.example.app.dto.TaskDto;
import com.example.app.dto.UserDto;
import org.springframework.stereotype.Component;

import java.util.stream.Collectors;

@Component
public class DtoMapper {

    public UserDto toDto(User user) {
        if (user == null) {
            return null;
        }

        UserDto dto = new UserDto();
        dto.setUserId(user.getId());
        dto.setUsername(user.getUsername());
        dto.setEmail(user.getEmail());
        dto.setFirstName(user.getFirstName());
        dto.setLastName(user.getLastName());
        dto.setRole(user.getRole());
        return dto;
    }

    public User toEntity(UserDto dto) {
        if (dto == null) {
            return null;
        }

        User user = new User();
        user.setId(dto.getUserId());
        user.setUsername(dto.getUsername());
        user.setEmail(dto.getEmail());
        user.setFirstName(dto.getFirstName());
        user.setLastName(dto.getLastName());
        user.setRole(dto.getRole());
        user.setPassword(dto.getPassword());
        return user;
    }

    public ProjectDto toDto(Project project) {
        if (project == null) {
            return null;
        }

        ProjectDto dto = new ProjectDto();
        dto.setProjectId(project.getId());
        dto.setName(project.getName());

        if (project.getUsers() != null) {
            dto.setUsers(project.getUsers().stream()
                    .map(this::toDto)
                    .collect(Collectors.toSet()));
        }

        return dto;
    }

    public Project toEntity(ProjectDto dto) {
        if (dto == null) {
            return null;
        }

        Project project = new Project();
        project.setId(dto.getProjectId());
        project.setName(dto.getName());
        return project;
    }

    public TaskDto toDto(Task task) {
        if (task == null) {
            return null;
        }

        TaskDto dto = new TaskDto();
        dto.setTaskId(task.getId());
        dto.setName(task.getName());
        dto.setPriority(task.getPriority());
        dto.setStatus(task.getStatus());
        dto.setAssignedUser(toDto(task.getAssignedUser()));
        dto.setProject(toDto(task.getProject()));
        return dto;
    }

    public Task toEntity(TaskDto dto) {
        if (dto == null) {
            return null;
        }

        Task task = new Task();
        task.setId(dto.getTaskId());
        task.setName(dto.getName());
        task.setPriority(dto.getPriority());
        task.setStatus(dto.getStatus());

        if (dto.getAssignedUser() != null && dto.getAssignedUser().getUserId() != null) {
            User user = new User();
            user.setId(dto.getAssignedUser().getUserId());
            task.setAssignedUser(user);
        }

        if (dto.getProject() != null && dto.getProject().getProjectId() != null) {
            Project project = new Project();
            project.setId(dto.getProject().getProjectId());
            task.setProject(project);
        }

        return task;
    }
}
