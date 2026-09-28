import { MessageSquarePlus, Trash2 } from "lucide-react";

import { AccountMenu } from "@/components/account-menu";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar";
import { useShop } from "@/context/shop-context";
import type { Conversation } from "@/types/ecommerce";

const DAY = 24 * 60 * 60 * 1000;

function bucketOf(createdAt: number): "Today" | "Yesterday" | "Earlier" {
  const age = Date.now() - createdAt;
  if (age < DAY) return "Today";
  if (age < 2 * DAY) return "Yesterday";
  return "Earlier";
}

function groupByRecency(conversations: Conversation[]) {
  const groups: { label: "Today" | "Yesterday" | "Earlier"; items: Conversation[] }[] = [
    { label: "Today", items: [] },
    { label: "Yesterday", items: [] },
    { label: "Earlier", items: [] },
  ];
  for (const conversation of conversations) {
    groups.find((group) => group.label === bucketOf(conversation.createdAt))?.items.push(conversation);
  }
  return groups.filter((group) => group.items.length > 0);
}

export function AppSidebar() {
  const { conversations, activeConversation, newChat, selectConversation, deleteConversation } = useShop();
  const groups = groupByRecency(conversations);

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <div className="flex items-center gap-3 px-2 py-3">
          <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-sidebar-accent text-sm font-semibold text-sidebar-accent-foreground">
            A
          </span>
          <span className="truncate text-base font-semibold text-sidebar-foreground group-data-[collapsible=icon]:hidden">
            Aether
          </span>
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton onClick={newChat} tooltip="New conversation">
                  <MessageSquarePlus />
                  <span>New chat</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {groups.map((group) => (
          <SidebarGroup key={group.label}>
            <SidebarGroupLabel>{group.label}</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {group.items.map((conversation) => (
                  <SidebarMenuItem key={conversation.id}>
                    <SidebarMenuButton
                      tooltip={conversation.title}
                      isActive={conversation.id === activeConversation.id}
                      onClick={() => selectConversation(conversation.id)}
                      className="pr-8"
                    >
                      <span className="truncate">{conversation.title}</span>
                    </SidebarMenuButton>
                    <SidebarMenuAction
                      showOnHover
                      aria-label={`Delete ${conversation.title}`}
                      onClick={() => deleteConversation(conversation.id)}
                    >
                      <Trash2 className="size-3" />
                    </SidebarMenuAction>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
      </SidebarContent>

      <SidebarFooter>
        <AccountMenu />
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  );
}
