package com.project.controller;

import com.project.model.User;
import com.project.service.UserService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class UserController {
    private final UserService userService;

    public UserController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping("/userListMock")
    public String oldUserListMock() {
        return "redirect:/userList";
    }

    @GetMapping("/userEditMock")
    public String oldUserEditMock(@RequestParam(name = "userId", required = false) Long userId) {
        return userId == null ? "redirect:/userEdit" : "redirect:/userEdit?userId=" + userId;
    }

    @GetMapping("/userList")
    public String userList(Model model) {
        model.addAttribute("users", userService.getAllUsers());
        return "userListMock";
    }

    @GetMapping("/userEdit")
    public String userEdit(@RequestParam(name = "userId", required = false) Long userId, Model model) {
        if (userId != null) {
            model.addAttribute("user", userService.getUserById(userId));
        } else {
            User user = new User();
            user.setRole("ROLE_USER");
            model.addAttribute("user", user);
        }
        return "userEditMock";
    }

    @PostMapping(path = "/userEdit", params = "cancel")
    public String userEditCancel() {
        return "redirect:/userList";
    }

    @PostMapping(path = "/userEdit", params = "delete")
    public String userDelete(@ModelAttribute("user") User user) {
        if (user.getUserId() != null) {
            userService.deleteUser(user.getUserId());
        }
        return "redirect:/userList";
    }

    @PostMapping("/userEdit")
    public String userSave(@ModelAttribute("user") User user) {
        if (user.getUserId() == null) {
            userService.createUser(user);
        } else {
            userService.updateUser(user.getUserId(), user);
        }
        return "redirect:/userList";
    }
}
