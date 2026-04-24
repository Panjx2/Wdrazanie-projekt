package com.project.controller;

import com.project.exception.HttpException;
import com.project.model.Project;
import com.project.model.User;
import com.project.service.ProjectService;
import com.project.service.TaskService;
import com.project.service.UserService;
import jakarta.validation.Valid;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class ProjectController {
    private final ProjectService projectService;
    private final TaskService taskService;
    private final UserService userService;

    public ProjectController(ProjectService projectService, TaskService taskService, UserService userService) {
        this.projectService = projectService;
        this.taskService = taskService;
        this.userService = userService;
    }

    @GetMapping("/")
    public String home() {
        return "redirect:/projektList";
    }

    @GetMapping("/projektList")
    public String projektList(Model model) {
        model.addAttribute("projekty", projectService.getAllProjects());
        return "projektList";
    }

    @GetMapping("/projektDetails")
    public String projektDetails(@RequestParam(name = "projektId", required = false) Long projektId, Model model) {
        if (projektId == null) {
            return "redirect:/projektList";
        }

        model.addAttribute("projekt", projectService.getProjectById(projektId));
        model.addAttribute("tasks", taskService.getTasksByProject(projektId));
        return "projektDetails";
    }

    @GetMapping("/projektEdit")
    public String projektEdit(@RequestParam(name = "projektId", required = false) Long projektId, Model model) {
        if (projektId != null) {
            model.addAttribute("projekt", projectService.getProjectById(projektId));
        } else {
            model.addAttribute("projekt", new Project());
        }
        return "projektEdit";
    }

    @PostMapping(path = "/projektEdit", params = "cancel")
    public String projektEditCancel() {
        return "redirect:/projektList";
    }

    @PostMapping(path = "/projektEdit", params = "delete")
    public String projektEditDelete(@ModelAttribute("projekt") Project projekt) {
        if (projekt.getProjectId() != null) {
            projectService.deleteProject(projekt.getProjectId());
        }
        return "redirect:/projektList";
    }

    @PostMapping(path = "/projektEdit")
    public String projektEditSave(@ModelAttribute("projekt") @Valid Project projekt, BindingResult bindingResult) {
        if (bindingResult.hasErrors()) {
            return "projektEdit";
        }
        try {
            if (projekt.getProjectId() == null) {
                projectService.createProject(projekt);
            } else {
                projectService.updateProject(projekt.getProjectId(), projekt);
            }
        } catch (HttpException e) {
            bindingResult.reject("http.error", e.getMessage());
            return "projektEdit";
        }
        return "redirect:/projektList";
    }

    @GetMapping("/projectStudentsAssign")
    public String projectStudentsAssign(@RequestParam(name = "projectId", required = false) Long projectId, Model model) {
        if (projectId == null) {
            return "redirect:/projektList";
        }

        Project project = projectService.getProjectById(projectId);
        List<User> users = userService.getAllUsers();
        Set<Long> assignedUserIds = project.getUsers() == null
                ? Collections.emptySet()
                : project.getUsers().stream().map(User::getUserId).collect(Collectors.toSet());

        model.addAttribute("projectId", projectId);
        model.addAttribute("projectName", project.getName());
        model.addAttribute("students", users);
        model.addAttribute("assignedStudentIds", assignedUserIds);
        return "projectStudentsAssign";
    }

    @PostMapping("/projectStudentsAssign")
    public String projectStudentsAssignSave(@RequestParam Long projectId,
                                            @RequestParam Long userId,
                                            @RequestParam String action) {
        if ("remove".equals(action)) {
            projectService.removeUserFromProject(projectId, userId);
        } else {
            projectService.addUserToProject(projectId, userId);
        }
        return "redirect:/projectStudentsAssign?projectId=" + projectId;
    }
}
