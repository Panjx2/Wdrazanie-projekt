package com.example.app.controller;

import com.example.app.data.TaskStatus;
import com.example.app.dto.TaskDto;
import com.example.app.mapper.DtoMapper;
import com.example.app.service.TaskService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/tasks")
public class TaskController {

    private final TaskService taskService;
    private final DtoMapper dtoMapper;

    public TaskController(TaskService taskService, DtoMapper dtoMapper) {
        this.taskService = taskService;
        this.dtoMapper = dtoMapper;
    }

    @PostMapping("/project/{projectId}")
    public TaskDto createTask(@PathVariable Long projectId,
                              @RequestBody TaskDto taskDto) {
        return dtoMapper.toDto(taskService.createTask(projectId, dtoMapper.toEntity(taskDto)));
    }

    @GetMapping("/{id}")
    public TaskDto getTask(@PathVariable Long id) {
        return dtoMapper.toDto(taskService.getTaskById(id));
    }

    @GetMapping("/project/{projectId}")
    public List<TaskDto> getTasksByProject(@PathVariable Long projectId) {
        return taskService.getTasksByProject(projectId).stream()
                .map(dtoMapper::toDto)
                .toList();
    }

    @PutMapping("/{taskId}")
    public TaskDto updateTask(@PathVariable Long taskId,
                              @RequestBody TaskDto taskDto) {
        return dtoMapper.toDto(taskService.updateTask(taskId, dtoMapper.toEntity(taskDto)));
    }

    @PatchMapping("/{taskId}/assign/{userId}")
    public TaskDto assignUser(@PathVariable Long taskId,
                              @PathVariable Long userId) {
        return dtoMapper.toDto(taskService.assignUser(taskId, userId));
    }

    @PatchMapping("/{taskId}/status")
    public TaskDto changeStatus(@PathVariable Long taskId,
                                @RequestParam TaskStatus status) {
        return dtoMapper.toDto(taskService.changeStatus(taskId, status));
    }

    @DeleteMapping("/{taskId}")
    public void deleteTask(@PathVariable Long taskId) {
        taskService.deleteTask(taskId);
    }
}
