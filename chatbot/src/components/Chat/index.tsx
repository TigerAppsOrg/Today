//Modules
import warning from "@/assets/warning.svg";
import { useRef, useEffect } from "react";
import { useChat } from "@/store/chat";
import { useForm } from "react-hook-form";
import { useAutoAnimate } from "@formkit/auto-animate/react";
import { OpenAIApi, Configuration } from "openai";
import { useMutation } from "react-query";
import TayAvatar from "@/assets/tayavatar.png";
import UserAvatar from "@/assets/useravatar.png";
import { useSearchParams } from "react-router-dom";

//Components
import { Input } from "@/components/Input";
import { FiSend, FiUser } from "react-icons/fi";
import { Avatar, IconButton, Spinner, Stack, Text } from "@chakra-ui/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Instructions } from "../Layout/Instructions";
import { useAPI } from "@/store/api";
import config from "@/config";

export interface ChatProps {}

interface ChatSchema {
  input: string;
}

export const Chat = ({ ...props }: ChatProps) => {
  const { api } = useAPI();
  const [searchParams, setSearchParams] = useSearchParams();
  const { selectedChat, addMessage, editMessage, addChat, editChat } =
    useChat();
  const selectedId = selectedChat?.id,
    selectedRole = selectedChat?.role;

  const hasSelectedChat = selectedChat && selectedChat?.content.length > 0;

  const { register, setValue, handleSubmit } = useForm<ChatSchema>();

  const [parentRef] = useAutoAnimate();

  const configuration = new Configuration({
    apiKey: api,
  });

  const openAi = new OpenAIApi(configuration);

  const { mutate, isLoading } = useMutation({
    mutationKey: "prompt",
    mutationFn: async (prompt: string) =>
      await openAi.createChatCompletion({
        model: "gpt-3.5-turbo",
        max_tokens: 256,
        messages: [{ role: "user", content: prompt }],
      }),
  });

  const handleAsk = async ({ input: prompt }: ChatSchema) => {
    const sendRequest = (
      selectedId: string,
      selectedChat: any
    ) => {
      setValue("input", "");

      addMessage(selectedId, {
        emitter: "user",
        message: prompt,
      });

      const controller = new AbortController();

      fetch(config.URL + "/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: prompt,
          uuid: searchParams.get("uuid"),
          session_id: selectedId,
        }),
        signal: controller.signal,
      })
        .then(async (response) => {
          const reader = response.body?.getReader();
          const decoder = new TextDecoder();
          let message = "";
          if (reader) {
            addMessage(selectedId, {
              emitter: "gpt",
              message,
            });
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              const chunk = decoder.decode(value, { stream: true });
              message += chunk;
              if (selectedChat) {
                editMessage(selectedId, message);
              }
            }

            if (selectedRole == "New chat" || selectedRole == undefined) {
              editChat(selectedId, { role: prompt });
            }
          }
        })
        .catch((error) => {
          console.error("Streaming error:", error);
          addMessage(selectedId, {
            emitter: "error",
            message: "An error occurred while processing the request.",
          });
        });
    };

    if (selectedId) {
      if (prompt && !isLoading) {
        sendRequest(selectedId, selectedChat);
      }
    } else {
      addChat(sendRequest);
    }
  };

  const AlwaysScrollToBottom = () => {
    const elementRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
      if (elementRef.current) {
        elementRef.current.scrollIntoView();
      }
    });
    return <div ref={elementRef} />;
  };

  const ExternalLink = ({ href, children }: any) => {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    );
  };

  useEffect(() => {
    const query = searchParams.get("query");
    if (query && query != "") {
      handleAsk({ input: query });
    }
  }, []);

  return (
    <Stack width="full" height="full" backgroundColor="#212529">
      <Stack
        maxWidth="768px"
        width="full"
        marginX="auto"
        height="82%"
        overflow="auto"
        backgroundColor="#212529"
      >
        <Stack spacing={2} padding={2} ref={parentRef} height="full">
          {hasSelectedChat ? (
            selectedChat.content.map(({ emitter, message }, key) => {
              const getAvatar = () => {
                switch (emitter) {
                  case "gpt":
                    return TayAvatar;
                  case "error":
                    return warning;
                  default:
                    return UserAvatar;
                }
              };

              const getMessage = () => {
                if (message.slice(0, 2) == "\n\n") {
                  return message.slice(2, Infinity);
                }

                return message;
              };

              return (
                <Stack
                  key={key}
                  direction="row"
                  padding={4}
                  rounded={8}
                  backgroundColor={emitter == "gpt" ? "#1e2022" : "transparent"}
                  spacing={4}
                >
                  <Avatar
                    name={emitter}
                    mt={2}
                    boxSize={"54px"}
                    src={getAvatar()}
                  />
                  <Text
                    whiteSpace="pre-wrap"
                    marginTop=".75em !important"
                    overflow="hidden"
                  >
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        a: ExternalLink,
                      }}
                    >
                      {getMessage()}
                    </ReactMarkdown>
                  </Text>
                </Stack>
              );
            })
          ) : (
            <Instructions onClick={(text) => setValue("input", text)} />
          )}
          <AlwaysScrollToBottom />
        </Stack>
      </Stack>
      <Stack
        height="18%"
        padding={4}
        backgroundColor="#151719"
        justifyContent="center"
        alignItems="center"
        overflow="hidden"
      >
        <Stack maxWidth="768px" width="100%">
          <Input
            autoFocus={true}
            variant="filled"
            inputRightAddon={
              <IconButton
                aria-label="send_button"
                icon={!isLoading ? <FiSend /> : <Spinner />}
                backgroundColor="transparent"
                onClick={handleSubmit(handleAsk)}
              />
            }
            {...register("input")}
            onSubmit={console.log}
            onKeyDown={(e) => {
              if (e.key == "Enter") {
                handleAsk({ input: e.currentTarget.value });
              }
            }}
            backgroundColor="whiteAlpha.200"
          />
          <Text textAlign="center" fontSize="sm" opacity={0.5}>
            ⚠️ Highly experimental. Responses may not be accurate. Not intended
            for academic use. Our goal is to make information accessible. Your
            feedback will help us improve.
          </Text>
        </Stack>
      </Stack>
    </Stack>
  );
};
