package com.example.app.controller;

import com.example.app.data.User;
import com.example.app.dto.UserDto;
import com.example.app.mapper.DtoMapper;
import com.example.app.service.UserService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/users")
public class UserController {

    private final UserService userService;
    private final DtoMapper dtoMapper;

    public UserController(UserService userService, DtoMapper dtoMapper) {
        this.userService = userService;
        this.dtoMapper = dtoMapper;
    }

    @PostMapping
    public UserDto createUser(@RequestBody UserDto userDto) {
        User user = dtoMapper.toEntity(userDto);
        return dtoMapper.toDto(userService.createUser(user));
    }

    @GetMapping
    public List<UserDto> getUsers() {
        return userService.getAllUsers().stream()
                .map(dtoMapper::toDto)
                .toList();
    }

    @GetMapping("/{id}")
    public UserDto getUser(@PathVariable Long id) {
        return dtoMapper.toDto(userService.getUserById(id));
    }

    @PutMapping("/{id}")
    public UserDto updateUser(@PathVariable Long id,
                              @RequestBody UserDto userDto) {
        User user = dtoMapper.toEntity(userDto);
        return dtoMapper.toDto(userService.updateUser(id, user));
    }

    @DeleteMapping("/{id}")
    public void deleteUser(@PathVariable Long id) {
        userService.deleteUser(id);
    }
}
