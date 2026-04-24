package com.project.controller;

import com.project.model.Project;
import com.project.model.Task;
import com.project.model.TaskStatus;
import com.project.model.User;
import com.project.service.ProjectService;
import com.project.service.TaskService;
import com.project.service.UserService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class TaskController {
    private final TaskService taskService;
    private final ProjectService projectService;
    private final UserService userService;

    public TaskController(TaskService taskService, ProjectService projectService, UserService userService) {
        this.taskService = taskService;
        this.projectService = projectService;
        this.userService = userService;
    }

    @GetMapping("/taskListMock")
    public String oldTaskListMock(@RequestParam(name = "projectId", required = false) Long projectId) {
        return projectId == null ? "redirect:/projektList" : "redirect:/taskList?projectId=" + projectId;
    }

    @GetMapping("/taskEditMock")
    public String oldTaskEditMock(@RequestParam(name = "projectId", required = false) Long projectId,
                                  @RequestParam(name = "taskId", required = false) Long taskId) {
        String redirect = "redirect:/taskEdit";
        if (projectId != null) {
            redirect += "?projectId=" + projectId;
            if (taskId != null) {
                redirect += "&taskId=" + taskId;
            }
        }
        return redirect;
    }

    @GetMapping("/taskList")
    public String taskList(@RequestParam(name = "projectId", required = false) Long projectId, Model model) {
        if (projectId == null) {
            return "redirect:/projektList";
        }

        model.addAttribute("project", projectService.getProjectById(projectId));
        model.addAttribute("projectId", projectId);
        model.addAttribute("tasks", taskService.getTasksByProject(projectId));
        return "taskListMock";
    }

    @GetMapping("/taskEdit")
    public String taskEdit(@RequestParam(name = "projectId", required = false) Long projectId,
                           @RequestParam(name = "taskId", required = false) Long taskId,
                           Model model) {
        if (projectId == null && taskId == null) {
            return "redirect:/projektList";
        }

        Task task;
        if (taskId != null) {
            task = taskService.getTaskById(taskId);
            if (task.getProject() != null) {
                projectId = task.getProject().getProjectId();
            }
            if (task.getAssignedUser() != null) {
                task.setAssignedUserId(task.getAssignedUser().getUserId());
            }
        } else {
            task = new Task();
            task.setStatus(TaskStatus.TODO);
        }
        task.setProjectId(projectId);

        model.addAttribute("task", task);
        model.addAttribute("projectId", projectId);
        model.addAttribute("users", userService.getAllUsers());
        model.addAttribute("statuses", TaskStatus.values());
        return "taskEditMock";
    }

    @PostMapping(path = "/taskEdit", params = "cancel")
    public String taskEditCancel(@ModelAttribute("task") Task task) {
        Long projectId = task.getProjectId();
        return projectId == null ? "redirect:/projektList" : "redirect:/taskList?projectId=" + projectId;
    }

    @PostMapping(path = "/taskEdit", params = "delete")
    public String taskDelete(@ModelAttribute("task") Task task) {
        if (task.getTaskId() != null) {
            taskService.deleteTask(task.getTaskId());
        }
        Long projectId = task.getProjectId();
        return projectId == null ? "redirect:/projektList" : "redirect:/taskList?projectId=" + projectId;
    }

    @PostMapping("/taskEdit")
    public String taskSave(@ModelAttribute("task") Task task) {
        Project project = new Project();
        project.setProjectId(task.getProjectId());
        task.setProject(project);

        if (task.getAssignedUserId() != null) {
            User user = new User();
            user.setUserId(task.getAssignedUserId());
            task.setAssignedUser(user);
        } else {
            task.setAssignedUser(null);
        }

        if (task.getTaskId() == null) {
            taskService.createTask(task.getProjectId(), task);
        } else {
            taskService.updateTask(task.getTaskId(), task);
        }

        return "redirect:/taskList?projectId=" + task.getProjectId();
    }

    @GetMapping("/taskStudentsAssign")
    public String taskStudentsAssign(@RequestParam(name = "taskId", required = false) Long taskId, Model model) {
        if (taskId == null) {
            return "redirect:/projektList";
        }

        Task task = taskService.getTaskById(taskId);
        model.addAttribute("task", task);
        model.addAttribute("taskId", taskId);
        model.addAttribute("taskTitle", task.getName());
        model.addAttribute("students", userService.getAllUsers());
        model.addAttribute("assignedUserId", task.getAssignedUser() == null ? null : task.getAssignedUser().getUserId());
        return "taskStudentsAssign";
    }

    @PostMapping("/taskStudentsAssign")
    public String taskStudentsAssignSave(@RequestParam Long taskId, @RequestParam Long userId) {
        Task task = taskService.assignUser(taskId, userId);
        Long projectId = task.getProject() == null ? null : task.getProject().getProjectId();
        return projectId == null ? "redirect:/projektList" : "redirect:/taskList?projectId=" + projectId;
    }
}
